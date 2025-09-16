import sys

sys.path.extend([".", ".."])
from CONSTANTS import *
from preprocessing.BasicLoader import BasicDataLoader


class Open5GSLoader(BasicDataLoader):
    def __init__(self, in_file=None,
                 dataset_base=os.path.join(PROJECT_ROOT, 'datasets/open5gs'),
                 semantic_repr_func=None):
        super(Open5GSLoader, self).__init__()

        self.logger = logging.getLogger('Open5GSLoader')
        self.logger.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        file_handler = logging.FileHandler(os.path.join(LOG_ROOT, 'Open5GSLoader.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - " + SESSION + " - %(levelname)s: %(message)s"))

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.info(
            'Construct self.logger success, current working directory: %s, logs will be written in %s' %
            (os.getcwd(), LOG_ROOT))

        self.dataset_base = dataset_base
        self.logs_root = os.path.join(self.dataset_base, 'logs')
        if in_file is None:
            in_file = os.path.join(self.dataset_base, 'open5gs.log')
        self.in_file = in_file
        self.semantic_repr_func = semantic_repr_func

        self._load_raw_log_seqs()

    def logger(self):
        return self.logger

    def _pre_process(self, line):
        return line.rstrip('\n')

    def _load_raw_log_seqs(self):
        sequence_file = os.path.join(self.dataset_base, 'raw_log_seqs.txt')
        label_file = os.path.join(self.dataset_base, 'label.txt')
        os.makedirs(self.dataset_base, exist_ok=True)

        if os.path.exists(sequence_file) and os.path.exists(label_file) and os.path.exists(self.in_file):
            self.logger.info('Start load from previous extraction. File path %s' % sequence_file)
            with open(sequence_file, 'r', encoding='utf-8') as reader:
                for line in tqdm(reader.readlines()):
                    tokens = line.strip().split(':')
                    block = tokens[0]
                    seq = tokens[1].split() if len(tokens) > 1 else []
                    if block not in self.block2seqs.keys():
                        self.block2seqs[block] = []
                        self.blocks.append(block)
                    self.block2seqs[block] = [int(x) for x in seq]
            with open(label_file, 'r', encoding='utf-8') as reader:
                for line in reader.readlines():
                    block_id, label = line.strip().split(':')
                    self.block2label[block_id] = label
        else:
            self.logger.info('Start loading Open5GS log sequences.')
            if not os.path.exists(self.logs_root):
                self.logger.error('Logs root %s not found.' % self.logs_root)
                exit(1)
            log_files = []
            for root, _, files in os.walk(self.logs_root):
                for file in files:
                    if file.endswith('.log'):
                        log_files.append(os.path.join(root, file))
            log_files.sort()
            if len(log_files) == 0:
                self.logger.error('No .log files found under %s' % self.logs_root)
                exit(1)
            os.makedirs(os.path.dirname(self.in_file), exist_ok=True)
            with open(self.in_file, 'w', encoding='utf-8') as writer:
                log_id = 0
                block_idx = 0
                for file_path in log_files:
                    block_id = str(block_idx)
                    self.blocks.append(block_id)
                    self.block2seqs[block_id] = []
                    self.block2label[block_id] = 'Normal'
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as reader:
                        for line in reader:
                            writer.write(line)
                            self.block2seqs[block_id].append(log_id)
                            log_id += 1
                    block_idx += 1

            with open(sequence_file, 'w', encoding='utf-8') as writer:
                for block in self.blocks:
                    writer.write(':'.join([block, ' '.join([str(x) for x in self.block2seqs[block]])]) + '\n')

            with open(label_file, 'w', encoding='utf-8') as writer:
                for block in self.block2label.keys():
                    writer.write(':'.join([block, self.block2label[block]]) + '\n')

        self.logger.info('Extraction finished successfully.')
        pass
