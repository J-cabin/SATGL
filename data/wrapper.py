
import os

from satgl.data.dataset.sat_dataset import MaxSATDataset
from satgl.data.dataloader.sat_dataloader import MaxSATDataLoader
from satgl.data.dataset.sat_dataset import SatistifiabilityDataset
from satgl.data.dataloader.sat_dataloader import SatistifiabilityDataLoader
from satgl.data.dataset.sat_dataset import UnSATCoreDataset
from satgl.data.dataloader.sat_dataloader import UnsatCoreDataLoader
from satgl.logger.logger import Logger


class SATDataWrapper():
    def __init__(self, config):
        self.config = config
        self.logger = Logger(config, name='sat_data_wrapper')
        self._load_dataset()
        self._build_dataloader()
        
    def _load_dataset(self):
        if self.config["task"] == "satisfiability":
            self._load_satisfiability_dataset()
        elif self.config["task"] == "maxsat":
            self._load_maxsat_dataset()
        elif self.config["task"] == "unsat_core":
            self._load_unsat_core_dataset()
        else:
            raise NotImplementedError(f"task {self.config['task']} not implemented.")
    
    def _build_dataloader(self):
        if self.config["task"] == "satisfiability":
            self._build_satisfiability_dataloader()
        elif self.config["task"] == "maxsat":
            self._build_maxsat_dataloader()
        elif self.config["task"] == "unsat_core":
            self._build_unsat_core_dataloader()
        else:
            raise NotImplementedError(f"task {self.config['task']} not implemented.")
    
    def _load_satisfiability_dataset(self): 
        if self.config["load_split_dataset"] == True:
            train_cnf_dir = os.path.join(self.config["dataset_path"], "train")
            valid_cnf_dir = os.path.join(self.config["dataset_path"], "valid")
            test_cnf_dir = os.path.join(self.config["dataset_path"], "test")
            train_label_path = os.path.join(self.config["dataset_path"], "label", "train.csv")
            valid_label_path = os.path.join(self.config["dataset_path"], "label", "valid.csv")
            test_label_path = os.path.join(self.config["dataset_path"], "label", "test.csv")

            self.logger.info("processing train dataset ...")
            self.train_dataset = SatistifiabilityDataset(self.config, train_cnf_dir, train_label_path)
            self.logger.info("processing valid dataset ...")
            self.valid_dataset = SatistifiabilityDataset(self.config, valid_cnf_dir, valid_label_path) if os.path.exists(valid_cnf_dir) else None
            self.logger.info("processing test dataset ...")
            self.test_dataset = SatistifiabilityDataset(self.config, test_cnf_dir, test_label_path) if os.path.exists(test_cnf_dir) else None
        else:
            raise NotImplementedError("todo ! Not implemented yet.")

    def _load_maxsat_dataset(self):
        if self.config["load_split_dataset"] == True:
            train_cnf_dir = os.path.join(self.config["dataset_path"], "train")
            valid_cnf_dir = os.path.join(self.config["dataset_path"], "valid")
            test_cnf_dir = os.path.join(self.config["dataset_path"], "test")
            train_label_path = os.path.join(self.config["dataset_path"], "label", "train.csv")
            valid_label_path = os.path.join(self.config["dataset_path"], "label", "valid.csv")
            test_label_path = os.path.join(self.config["dataset_path"], "label", "test.csv")

            self.logger.info("processing train dataset ...")
            self.train_dataset = MaxSATDataset(self.config, train_cnf_dir, train_label_path)
            self.logger.info("processing valid dataset ...")
            self.valid_dataset = MaxSATDataset(self.config, valid_cnf_dir,
                                                         valid_label_path) if os.path.exists(valid_cnf_dir) else None
            self.logger.info("processing test dataset ...")
            self.test_dataset = MaxSATDataset(self.config, test_cnf_dir, test_label_path) if os.path.exists(
                test_cnf_dir) else None
        else:
            raise NotImplementedError("todo ! Not implemented yet.")

    def _load_unsat_core_dataset(self):
        if self.config["load_split_dataset"] == True:
            train_cnf_dir = os.path.join(self.config["dataset_path"], "train")
            valid_cnf_dir = os.path.join(self.config["dataset_path"], "valid")
            test_cnf_dir = os.path.join(self.config["dataset_path"], "test")
            train_label_path = os.path.join(self.config["dataset_path"], "label", "train.csv")
            valid_label_path = os.path.join(self.config["dataset_path"], "label", "valid.csv")
            test_label_path = os.path.join(self.config["dataset_path"], "label", "test.csv")

            self.logger.info("processing train dataset ...")
            self.train_dataset = UnSATCoreDataset(self.config, train_cnf_dir, train_label_path)
            self.logger.info("processing valid dataset ...")
            self.valid_dataset = UnSATCoreDataset(self.config, valid_cnf_dir,
                                               valid_label_path) if os.path.exists(valid_cnf_dir) else None
            self.logger.info("processing test dataset ...")
            self.test_dataset = UnSATCoreDataset(self.config, test_cnf_dir, test_label_path) if os.path.exists(
                test_cnf_dir) else None
        else:
            raise NotImplementedError("Not implemented yet.")

    def _build_satisfiability_dataloader(self):
        pair_wise_batch_size = max(self.config["batch_size"] // 2, 1)
        self.train_dataloader = SatistifiabilityDataLoader(self.train_dataset, pair_wise_batch_size, shuffle=True)
        self.valid_dataloader = SatistifiabilityDataLoader(self.valid_dataset, pair_wise_batch_size, shuffle=False) if self.valid_dataset is not None else None
        self.test_dataloader = SatistifiabilityDataLoader(self.test_dataset, pair_wise_batch_size, shuffle=False) if self.test_dataset is not None else None


    def _build_maxsat_dataloader(self):
        batch_size = self.config["batch_size"]
        self.train_dataloader = MaxSATDataLoader(self.train_dataset, batch_size, shuffle=True)
        self.valid_dataloader = MaxSATDataLoader(self.valid_dataset, batch_size, shuffle=False) if self.valid_dataset is not None else None
        self.test_dataloader = MaxSATDataLoader(self.test_dataset, batch_size, shuffle=False) if self.test_dataset is not None else None

    def _build_unsat_core_dataloader(self):
        batch_size = self.config["batch_size"]
        self.train_dataloader = UnsatCoreDataLoader(self.train_dataset, batch_size, shuffle=True)
        self.valid_dataloader = UnsatCoreDataLoader(self.valid_dataset, batch_size, shuffle=False) if self.valid_dataset is not None else None
        self.test_dataloader = UnsatCoreDataLoader(self.test_dataset, batch_size, shuffle=False) if self.test_dataset is not None else None
