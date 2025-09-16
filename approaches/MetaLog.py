import sys
import json
sys.path.extend([".", ".."])
from CONSTANTS import *
from sklearn.decomposition import FastICA
from representations.templates.statistics import Simple_template_TF_IDF, Template_TF_IDF_without_clean
from representations.sequences.statistics import Sequential_TF
from preprocessing.datacutter.SimpleCutting import cut_by_217, cut_by_316, cut_by_415, cut_by_514, cut_by_316_filter, cut_by_415_filter, cut_by_226_filter, cut_by_514_filter, cut_by_613_filter, cut_all
from preprocessing.AutoLabeling import Probabilistic_Labeling
from preprocessing.Preprocess import Preprocessor
from preprocessing.dataloader.BGLLoader import BGLLoader
from preprocessing.dataloader.HDFSLoader import HDFSLoader
from preprocessing.dataloader.OSLoader import OSLoader
from preprocessing.dataloader.Open5GSLoader import Open5GSLoader
from module.Optimizer import Optimizer
from module.Common import data_iter, generate_tinsts_binary_label, batch_variable_inst
from models.gru import AttGRUModel
from utils.Vocab import Vocab
from entities.instances import Instance
from entities.TensorInstances import TInstWithoutLogits
from parsers.Drain_IBM import Drain3Parser


processor_target = None
processor_source = None
vocab_target = None
vocab_source = None


def normalize_dataset_name(name):
    lower = name.lower()
    if lower == 'open5gs':
        return 'open5gs'
    if lower == 'openstack':
        return 'OpenStack'
    if lower == 'bglsample':
        return 'BGLSample'
    return name.upper()


def get_parser_config(dataset):
    if dataset == 'HDFS':
        return os.path.join(PROJECT_ROOT, 'conf/HDFS.ini')
    if dataset == 'BGL' or dataset == 'BGLSample':
        return os.path.join(PROJECT_ROOT, 'conf/BGL.ini')
    if dataset == 'OpenStack':
        return os.path.join(PROJECT_ROOT, 'conf/OpenStack.ini')
    if dataset.lower() == 'open5gs':
        return os.path.join(PROJECT_ROOT, 'conf/Open5GS.ini')
    raise ValueError('Unsupported dataset %s' % dataset)


def get_target_cut_func(dataset):
    upper = dataset.upper() if isinstance(dataset, str) else dataset
    if upper == 'BGL':
        return cut_by_316_filter
    if upper == 'HDFS':
        return cut_by_415
    if str(dataset).lower() == 'open5gs':
        return cut_by_316_filter
    if upper == 'OPENSTACK':
        return cut_by_316_filter
    return cut_by_316_filter


def get_source_cut_func(dataset):
    upper = dataset.upper() if isinstance(dataset, str) else dataset
    if upper == 'HDFS':
        return cut_by_415
    if upper == 'BGL':
        return cut_all
    if str(dataset).lower() == 'open5gs':
        return cut_by_316_filter
    if upper == 'OPENSTACK':
        return cut_all
    return cut_by_415


def merge_embeddings(primary_embedding, secondary_embedding):
    new_embedding = {}
    offset = 0
    for key in primary_embedding.keys():
        new_embedding[key] = primary_embedding[key]
    if new_embedding:
        offset = max(new_embedding.keys()) + 1
    for key in secondary_embedding.keys():
        new_embedding[key + offset] = secondary_embedding[key]
    return new_embedding


def build_loader_for_dataset(dataset):
    if dataset == 'HDFS':
        return HDFSLoader(in_file=os.path.join(PROJECT_ROOT, 'datasets/HDFS/HDFS.log'))
    if dataset == 'BGL' or dataset == 'BGLSample':
        dataset_base = os.path.join(PROJECT_ROOT, 'datasets/' + dataset)
        in_file = os.path.join(dataset_base, dataset + '.log')
        return BGLLoader(in_file=in_file, dataset_base=dataset_base)
    if dataset == 'OpenStack':
        return OSLoader(in_file=os.path.join(PROJECT_ROOT, 'datasets/OpenStack/openstack_normal1.log'),
                        ab_in_file=os.path.join(PROJECT_ROOT, 'datasets/OpenStack/openstack_abnormal.log'))
    if dataset.lower() == 'open5gs':
        return Open5GSLoader(in_file=os.path.join(PROJECT_ROOT, 'datasets/open5gs/open5gs.log'))
    raise ValueError('Unsupported dataset %s' % dataset)


def build_inference_tinst(instances, vocab):
    if not instances:
        return None
    slen = 0
    for inst in instances:
        if len(inst.sequence) > slen:
            slen = len(inst.sequence)
    if slen == 0:
        slen = 1
    slen = min(slen, 500)
    tinst = TInstWithoutLogits(len(instances), slen, 2)
    for b, inst in enumerate(instances):
        cur_len = min(len(inst.sequence), slen)
        tinst.word_len[b] = cur_len
        for index in range(cur_len):
            tinst.src_words[b, index] = vocab.word2id(inst.sequence[index])
            tinst.src_masks[b, index] = 1
    return tinst


lstm_hiddens = 100
num_layer = 2
batch_size = 100
epochs = 10


def get_updated_network(old, new, lr, load=False):
    updated_theta = {}
    state_dicts = old.state_dict()
    param_dicts = dict(old.named_parameters())

    for i, (k, v) in enumerate(state_dicts.items()):
        if k in param_dicts.keys() and param_dicts[k].grad is not None:
            updated_theta[k] = param_dicts[k] - lr * param_dicts[k].grad
        else:
            updated_theta[k] = state_dicts[k]
    if load:
        new.load_state_dict(updated_theta)
    else:
        new = put_theta(new, updated_theta)
    return new


def put_theta(model, theta):
    def k_param_fn(tmp_model, name=None):
        if len(tmp_model._modules) != 0:
            for (k, v) in tmp_model._modules.items():
                if name is None:
                    k_param_fn(v, name=str(k))
                else:
                    k_param_fn(v, name=str(name + '.' + k))
        else:
            for (k, v) in tmp_model._parameters.items():
                if not isinstance(v, torch.Tensor):
                    continue
                tmp_model._parameters[k] = theta[str(name + '.' + k)]

    k_param_fn(model)
    return model


class MetaLog:
    _logger = logging.getLogger('MetaLog')
    _logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
    file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'MetaLog.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))
    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    _logger.info(
        'Construct logger for MetaLog succeeded, current working directory: %s, logs will be written in %s' %
        (os.getcwd(), LOG_ROOT))

    @property
    def logger(self):
        return MetaLog._logger

    def __init__(self, vocab, num_layer, hidden_size, label2id):
        self.label2id = label2id
        self.vocab = vocab
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.batch_size = 128
        self.test_batch_size = 1024
        self.model = AttGRUModel(vocab, self.num_layer, self.hidden_size)
        self.bk_model = AttGRUModel(vocab, self.num_layer, self.hidden_size)
        if torch.cuda.is_available():
            self.model = self.model.cuda(device)
            self.bk_model = self.bk_model.cuda(device)
        self.loss = nn.BCELoss()

    def forward(self, inputs, targets):
        tag_logits = self.model(inputs)
        tag_logits = F.softmax(tag_logits, dim=1)
        loss = self.loss(tag_logits, targets)
        return loss

    def bk_forward(self, inputs, targets):
        tag_logits = self.bk_model(inputs)
        tag_logits = F.softmax(tag_logits, dim=1)
        loss = self.loss(tag_logits, targets)
        return loss

    def predict(self, inputs, threshold=None):
        with torch.no_grad():
            tag_logits = self.model(inputs)
            tag_logits = F.softmax(tag_logits)
        if threshold is not None:
            probs = tag_logits.detach().cpu().numpy()
            anomaly_id = self.label2id['Anomalous']
            pred_tags = np.zeros(probs.shape[0])
            for i, logits in enumerate(probs):
                if logits[anomaly_id] >= threshold:
                    pred_tags[i] = anomaly_id
                else:
                    pred_tags[i] = 1 - anomaly_id

        else:
            pred_tags = tag_logits.detach().max(1)[1].cpu()
        return pred_tags, tag_logits

    def evaluate(self, instances, threshold=0.5):
        self.logger.info('Start evaluating by threshold %.3f' % threshold)
        with torch.no_grad():
            self.model.eval()
            globalBatchNum = 0
            TP, TN, FP, FN = 0, 0, 0, 0
            tag_correct, tag_total = 0, 0
            for onebatch in data_iter(instances, self.test_batch_size, False):
                tinst = generate_tinsts_binary_label(onebatch, vocab_target, False)
                tinst.to_cuda(device)
                self.model.eval()
                pred_tags, tag_logits = self.predict(tinst.inputs, threshold)
                for inst, bmatch in batch_variable_inst(onebatch, pred_tags, tag_logits, processor_target.id2tag):
                    tag_total += 1
                    if bmatch:
                        tag_correct += 1
                        if inst.label == 'Normal':
                            TN += 1
                        else:
                            TP += 1
                    else:
                        if inst.label == 'Normal':
                            FP += 1
                        else:
                            FN += 1
                globalBatchNum += 1
            self.logger.info('TP: %d, TN: %d, FN: %d, FP: %d' % (TP, TN, FN, FP))
            if TP + FP != 0:
                precision = 100 * TP / (TP + FP)
                recall = 100 * TP / (TP + FN)
                f = 2 * precision * recall / (precision + recall)
                fpr = 100 * FP /  (FP + TN)
                self.logger.info('Precision = %d / %d = %.4f, Recall = %d / %d = %.4f F1 score = %.4f, FPR = %.4f'
                                 % (TP, (TP + FP), precision, TP, (TP + FN), recall, f, fpr))
            else:
                self.logger.info('Precision is 0 and therefore f is 0')
                precision, recall, f = 0, 0, 0
        return precision, recall, f


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', default='train', type=str, help='train, test or inference')
    argparser.add_argument('--parser', default='IBM', type=str,
                           help='Select parser, please see parser list for detail. Default Official.')
    argparser.add_argument('--min_cluster_size', type=int, default=100,
                           help="min_cluster_size.")
    argparser.add_argument('--min_samples', type=int, default=100,
                           help="min_samples")
    argparser.add_argument('--reduce_dimension', type=int, default=50,
                           help="Reduce dimentsion for fastICA, to accelerate the HDBSCAN probabilistic label estimation.")
    argparser.add_argument('--threshold', type=float, default=0.5,
                           help="Anomaly threshold.")
    argparser.add_argument('--beta', type=float, default=1.0,
                           help="weight for meta testing")
    argparser.add_argument('--target_dataset', default='BGL', type=str,
                           choices=['BGL', 'HDFS', 'OpenStack', 'open5gs', 'BGLSample'],
                           help='Target dataset for meta testing or inference.')
    argparser.add_argument('--source_dataset', default='HDFS', type=str,
                           choices=['HDFS', 'BGL', 'OpenStack', 'open5gs', 'BGLSample'],
                           help='Source dataset for meta training.')
    argparser.add_argument('--target_zero_label', action='store_true',
                           help='Skip target supervised fine-tuning when no labeled target data available.')
    argparser.add_argument('--inference_file', type=str, default=None,
                           help='Log file path for inference mode.')

    args, extra_args = argparser.parse_known_args()

    parser = args.parser
    mode = args.mode
    min_cluster_size = args.min_cluster_size
    min_samples = args.min_samples
    reduce_dimension = args.reduce_dimension
    threshold = args.threshold
    beta = args.beta
    target_dataset = normalize_dataset_name(args.target_dataset)
    source_dataset = normalize_dataset_name(args.source_dataset)
    target_zero_label = args.target_zero_label
    inference_file = args.inference_file

    save_dir = os.path.join(PROJECT_ROOT, 'outputs')
    dataset_output_root = os.path.join(save_dir, target_dataset) if target_dataset.lower() == 'open5gs' else save_dir
    dataset_identifier = target_dataset + '_' + parser
    base_result_dir = os.path.join(dataset_output_root, 'results/MetaLog', dataset_identifier)
    prob_label_res_file_target = os.path.join(base_result_dir,
                                              'prob_label_res',
                                              'mcs-' + str(min_cluster_size) + '_ms-' + str(min_samples))
    rand_state_target = os.path.join(base_result_dir, 'prob_label_res', 'random_state')
    output_model_dir = os.path.join(dataset_output_root, 'models/MetaLog', dataset_identifier, 'model')
    output_res_dir = os.path.join(base_result_dir, 'detect_res')
    inference_base_dir = os.path.join(dataset_output_root, 'inference', dataset_identifier)
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    if not os.path.exists(output_res_dir):
        os.makedirs(output_res_dir)

    template_encoder_target = Template_TF_IDF_without_clean() if target_dataset == 'NC' else Simple_template_TF_IDF()
    processor_target = Preprocessor()
    target_cut = get_target_cut_func(target_dataset)
    train_target, _, test_target = processor_target.process(dataset=target_dataset, parsing=parser,
                                                            cut_func=target_cut,
                                                            template_encoding=template_encoder_target.present)

    sequential_encoder_target = Sequential_TF(processor_target.embedding)
    train_reprs_target = sequential_encoder_target.present(train_target)
    for index, inst in enumerate(train_target):
        inst.repr = train_reprs_target[index]
    test_reprs_target = []
    if test_target:
        test_reprs_target = sequential_encoder_target.present(test_target)
        for index, inst in enumerate(test_target):
            inst.repr = test_reprs_target[index]

    transformer_target = None
    if reduce_dimension != -1 and train_target:
        start_time = time.time()
        print("Start FastICA, target dimension: %d" % reduce_dimension)
        transformer_target = FastICA(n_components=reduce_dimension)
        reduced_train = transformer_target.fit_transform(train_reprs_target)
        for idx, inst in enumerate(train_target):
            inst.repr = reduced_train[idx]
        if test_target:
            reduced_test = transformer_target.transform(test_reprs_target)
            for idx, inst in enumerate(test_target):
                inst.repr = reduced_test[idx]
        print('Finished at %.2f' % (time.time() - start_time))

    labeled_train_target = []
    if not target_zero_label and mode != 'inference':
        train_normal_target = [x for x, inst in enumerate(train_target) if inst.label == 'Normal']
        normal_ids_target = train_normal_target[:int(0.5 * len(train_normal_target))]
        label_generator_target = Probabilistic_Labeling(min_samples=min_samples, min_clust_size=min_cluster_size,
                                                       res_file=prob_label_res_file_target,
                                                       rand_state_file=rand_state_target)
        labeled_train_target = label_generator_target.auto_label(train_target, normal_ids_target)
        TP, TN, FP, FN = 0, 0, 0, 0
        for inst in labeled_train_target:
            if inst.predicted == 'Normal':
                if inst.label == 'Normal':
                    TN += 1
                else:
                    FN += 1
            else:
                if inst.label == 'Anomalous':
                    TP += 1
                else:
                    FP += 1
        from utils.common import get_precision_recall

        print(len(normal_ids_target))
        print('TP %d TN %d FP %d FN %d' % (TP, TN, FP, FN))
        p, r, f = get_precision_recall(TP, TN, FP, FN)
        print('%.4f, %.4f, %.4f' % (p, r, f))

    vocab_target = Vocab()
    vocab_target.load_from_dict(processor_target.embedding)

    template_encoder_source = Template_TF_IDF_without_clean() if source_dataset == 'NC' else Simple_template_TF_IDF()
    processor_source = Preprocessor()
    source_cut = get_source_cut_func(source_dataset)
    train_source, _, _ = processor_source.process(dataset=source_dataset, parsing=parser,
                                                  cut_func=source_cut,
                                                  template_encoding=template_encoder_source.present)

    sequential_encoder_source = Sequential_TF(processor_source.embedding)
    train_reprs_source = sequential_encoder_source.present(train_source)
    for index, inst in enumerate(train_source):
        inst.repr = train_reprs_source[index]

    transformer_source = None
    if reduce_dimension != -1 and train_source:
        start_time = time.time()
        print("Start FastICA, target dimension: %d" % reduce_dimension)
        transformer_source = FastICA(n_components=reduce_dimension)
        reduced_source = transformer_source.fit_transform(train_reprs_source)
        for idx, inst in enumerate(train_source):
            inst.repr = reduced_source[idx]
        print('Finished at %.2f' % (time.time() - start_time))

    labeled_train_source = train_source
    vocab_source = Vocab()
    vocab_source.load_from_dict(processor_source.embedding)

    new_embedding = merge_embeddings(processor_target.embedding, processor_source.embedding)
    vocab = Vocab()
    vocab.load_from_dict(new_embedding)

    metalog = MetaLog(vocab, num_layer, lstm_hiddens, processor_target.label2id)

    log = 'layer={}_hidden={}_epoch={}'.format(num_layer, lstm_hiddens, epochs)
    best_model_file = os.path.join(output_model_dir, log + '_best.pt')
    last_model_file = os.path.join(output_model_dir, log + '_last.pt')

    if mode == 'train':
        optimizer = Optimizer(filter(lambda p: p.requires_grad, metalog.model.parameters()), lr=2e-3)
        global_step = 0
        bestF = 0
        for epoch in range(epochs):
            metalog.model.train()
            metalog.bk_model.train()
            start = time.strftime("%H:%M:%S")
            metalog.logger.info("Starting epoch: %d | phase: train | start time: %s | learning rate: %s" %
                               (epoch, start, optimizer.lr))

            if not target_zero_label and labeled_train_target:
                batch_num = int(np.ceil(len(labeled_train_source) / float(batch_size)))
                batch_iter = 0
                batch_num_test = int(np.ceil(len(labeled_train_target) / float(batch_size)))
                batch_iter_test = 0
                total_bn = max(batch_num, batch_num_test)
                meta_train_loader = data_iter(labeled_train_source, batch_size, True)
                meta_test_loader = data_iter(labeled_train_target, batch_size, True)

                for i in range(total_bn):
                    optimizer.zero_grad()
                    meta_train_batch = meta_train_loader.__next__()
                    tinst_tr = generate_tinsts_binary_label(meta_train_batch, vocab_source)
                    tinst_tr.to_cuda(device)
                    loss = metalog.forward(tinst_tr.inputs, tinst_tr.targets)
                    loss_value = loss.data.cpu().numpy()
                    loss.backward(retain_graph=True)
                    batch_iter += 1
                    metalog.bk_model = get_updated_network(metalog.model, metalog.bk_model, 2e-3).train().cuda()

                    meta_test_batch = meta_test_loader.__next__()
                    tinst_test = generate_tinsts_binary_label(meta_test_batch, vocab_target)
                    tinst_test.to_cuda(device)
                    loss_te = beta * metalog.bk_forward(tinst_test.inputs, tinst_test.targets)
                    loss_value_te = loss_te.data.cpu().numpy() / beta
                    loss_te.backward()
                    batch_iter_test += 1

                    optimizer.step()
                    global_step += 1
                    if global_step % 500 == 0:
                        metalog.logger.info("Step:%d, Epoch:%d, meta train loss:%.2f, meta test loss:%.2f" %
                                           (global_step, epoch, loss_value, loss_value_te))
                    if batch_iter == batch_num:
                        meta_train_loader = data_iter(labeled_train_source, batch_size, True)
                        batch_iter = 0
                    if batch_iter_test == batch_num_test:
                        meta_test_loader = data_iter(labeled_train_target, batch_size, True)
                        batch_iter_test = 0
            else:
                for meta_train_batch in data_iter(labeled_train_source, batch_size, True):
                    optimizer.zero_grad()
                    tinst_tr = generate_tinsts_binary_label(meta_train_batch, vocab_source)
                    tinst_tr.to_cuda(device)
                    loss = metalog.forward(tinst_tr.inputs, tinst_tr.targets)
                    loss_value = loss.data.cpu().numpy()
                    loss.backward()
                    optimizer.step()
                    global_step += 1
                    if global_step % 500 == 0:
                        metalog.logger.info("Step:%d, Epoch:%d, meta train loss:%.2f" %
                                           (global_step, epoch, loss_value))

            if test_target and not target_zero_label:
                metalog.logger.info('Testing on test set.')
                _, _, f = metalog.evaluate(test_target, threshold)
                if f > bestF:
                    metalog.logger.info("Exceed best f: history = %.2f, current = %.2f" % (bestF, f))
                    torch.save(metalog.model.state_dict(), best_model_file)
                    bestF = f
            metalog.logger.info('Training epoch %d finished.' % epoch)
            torch.save(metalog.model.state_dict(), last_model_file)

    if mode == 'inference':
        checkpoint_path = best_model_file if os.path.exists(best_model_file) else last_model_file
        if not os.path.exists(checkpoint_path):
            metalog.logger.error('Model checkpoint not found. Expected at %s', checkpoint_path)
            sys.exit(1)
        if not inference_file:
            metalog.logger.error('Inference mode requires --inference_file.')
            sys.exit(1)
        metalog.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        metalog.model.eval()
        loader = build_loader_for_dataset(target_dataset)
        parser_config_file = get_parser_config(target_dataset)
        persistence_folder = os.path.join(PROJECT_ROOT, 'datasets', target_dataset, 'persistences')
        drain_parser = Drain3Parser(config_file=parser_config_file, persistence_folder=persistence_folder)
        if drain_parser.to_update:
            metalog.logger.error('Trained parser not found for dataset %s. Please run training pipeline first.',
                                 target_dataset)
            sys.exit(1)
        sequence = []
        with open(inference_file, 'r', encoding='utf-8', errors='ignore') as reader:
            for line in reader:
                processed_line = loader._pre_process(line)
                cluster = drain_parser.match(processed_line)
                if cluster is None:
                    event = processed_line
                else:
                    event = cluster.cluster_id
                sequence.append(event)
        inference_instances = [Instance('inference_0', sequence, 'Normal')]
        tinst = build_inference_tinst(inference_instances, vocab_target)
        if tinst is None:
            metalog.logger.error('No valid sequence constructed from inference file %s', inference_file)
            sys.exit(1)
        tinst.to_cuda(device)
        pred_tags, tag_logits = metalog.predict(tinst.inputs, threshold)
        anomaly_id = metalog.label2id['Anomalous']
        probs = tag_logits[:, anomaly_id].detach().cpu().numpy()
        if isinstance(pred_tags, np.ndarray):
            pred_indices = pred_tags.astype(int).tolist()
        else:
            pred_indices = pred_tags.detach().cpu().numpy().tolist()
        pred_labels = [processor_target.id2tag[int(idx)] for idx in pred_indices]
        run_dir = os.path.join(inference_base_dir, log)
        os.makedirs(run_dir, exist_ok=True)
        output_csv = os.path.join(run_dir, 'report.csv')
        with open(output_csv, 'w', encoding='utf-8') as writer:
            writer.write('sequence_id,anomaly_probability,prediction\n')
            for idx, (prob, label) in enumerate(zip(probs, pred_labels)):
                writer.write('%d,%.6f,%s\n' % (idx, prob, label))
        metadata_path = os.path.join(run_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as writer:
            json.dump({'threshold': threshold,
                       'note': 'Predictions flagged as Anomalous when probability >= threshold.'}, writer, indent=2)
        metalog.logger.info('Inference results written to %s', output_csv)
        sys.exit(0)

    if os.path.exists(last_model_file) and mode != 'inference':
        metalog.logger.info('=== Final Model ===')
        metalog.model.load_state_dict(torch.load(last_model_file, map_location=device))
        if test_target and not target_zero_label:
            metalog.evaluate(test_target, threshold)
    if os.path.exists(best_model_file) and not target_zero_label and test_target and mode != 'inference':
        metalog.logger.info('=== Best Model ===')
        metalog.model.load_state_dict(torch.load(best_model_file, map_location=device))
        metalog.evaluate(test_target, threshold)
    if mode != 'inference':
        metalog.logger.info('All Finished')
