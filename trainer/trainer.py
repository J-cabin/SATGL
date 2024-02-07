import os
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import (
        ReduceLROnPlateau,
        StepLR,
        ExponentialLR,
)
from satgl.logger.logger import Logger
from satgl.model.loss import get_loss
from satgl.trainer.optimizer import get_optimizer
from satgl.metric.metric import (
    eval_reduce,
    eval_all_metric,
    eval_compare
)
from torch.utils.tensorboard import SummaryWriter   
from copy import deepcopy

class AbstractTrainer(object):
    def __init__(self, config, model):
        self.config = config
        self.device = config.device
        self.model = model.to(config.device)
        self.logger = Logger(config, name='trainer')
        self.loss = get_loss(config=config)
        self.tensorboard_writer = SummaryWriter(config["tensorboard_dir"])
        self._set_optimizer()
        self._set_scheduler()
    
    def pred_fn(self, batched_data):
        raise NotImplementedError

    def label_fn(self, batched_data):
        raise NotImplementedError

    def _set_optimizer(self):
        self.optimizer = get_optimizer(config=self.config)(
            params=self.model.parameters(),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay)
        )
    
    def _set_scheduler(self):
        if self.config["scheduler_settings"] == False:
            self.scheduler = None
        else:
            if self.config["scheduler_settings"]["scheduler"] == "ReduceLROnPlateau":
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode=self.config["scheduler_settings"]["mode"],
                    factor=self.config["scheduler_settings"]["factor"],
                    patience=self.config["scheduler_settings"]["patience"],
                )
            elif self.config["scheduler_settings"]["scheduler"] == "StepLR":
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=self.config["scheduler_settings"]["step_size"],
                    gamma=self.config["scheduler_settings"]["gamma"]
                )
            else:
                raise NotImplementedError("scheduler not supportet")

    @torch.no_grad()
    def evaluate(self, 
                 eval_loader, 
                 **kwargs):
        """
            evaluate model on eval_loader
            Args:
                eval_loader: dataloader for evaluation
                kwargs:
                    pred_fn: function to get prediction from batched_data
                    label_fn: function to get label from batched_data
                    eval_fn: function to evaluate prediction and label
        """
        self.model.eval()
        result_dict = None
        iter_data = (
            tqdm(
                eval_loader,
                total=len(eval_loader),
                ncols=100,
                desc=f"eval "
            )
        )
        if "loss_fn" in kwargs:
            loss_fn = kwargs["loss_fn"]
        else:
            loss_fn = get_loss(config=self.config)
        if "pred_fn" in kwargs:
            pred_fn = kwargs["pred_fn"]
        else:
            pred_fn = self.pred_fn
        if "label_fn" in kwargs:
            label_fn = kwargs["label_fn"]
        else:
            label_fn = self.label_fn
        
        sum_loss = 0
        for batch_idx, batched_data in enumerate(iter_data):
            batched_pred = pred_fn(batched_data)
            batched_label = label_fn(batched_data)
            loss = loss_fn(batched_pred, batched_label)
            sum_loss += loss.item() * batched_pred.shape[0]

            if "eval_fn" in kwargs:
                cur_result_dict = kwargs["eval_fn"](pred=batched_pred, label=batched_label)
            else:
                cur_result_dict = eval_all_metric(config=self.config, pred=batched_pred, label=batched_label)

            result_dict = eval_reduce([result_dict, cur_result_dict])
            iter_data.set_postfix(**result_dict)
        
        result_dict["loss"] = sum_loss / result_dict["data_size"]
        
        return result_dict
    
    def scheduler_step(self, valid_result):
        if self.scheduler is None:
            return
        if self.config["scheduler_settings"]["scheduler"] == "ReduceLROnPlateau":
            self.scheduler.step(valid_result["loss"])
        else:
            raise NotImplementedError("todo other scheduler")

    def train(self, 
              train_loader, 
              valid_loader=None,
              test_loader=None,
              load_best_model=True,
              **kwargs):
        """
            train model on train_loader
            Args:
                train_loader: dataloader for training
                valid_loader: dataloader for validation
                test_loader: dataloader for testing
                kwargs:
                    loss_fn: function to calculate loss
                    pred_fn: function to get prediction from batched_data
                    label_fn: function to get label from batched_data
                    eval_fn: function to evaluate prediction and label
                    load_best_model: whether to load best model
                    
        """
        valid_metric = self.config["valid_metric"]
        best_result = None
        best_epoch = 0
        best_state_dict = None
        early_stop = self.config["early_stop"]

        for epoch_idx in range(self.config.epochs):
            train_result = self._train_epoch(train_loader=train_loader, **kwargs)

            valid_result = None
            # evaluate model every eval_step epochs
            if (epoch_idx + 1) % self.config.eval_step == 0 and valid_loader is not None:
                valid_result = self.evaluate(valid_loader, **kwargs)
                
            # tensorboard log
            self.tensorboard_writer.add_scalar("train/loss", train_result["loss"], epoch_idx)
            if valid_metric != "loss":
                self.tensorboard_writer.add_scalar(f"train/{valid_metric}", train_result[valid_metric], epoch_idx)
            
            # output result
            self.logger.info(f"epoch [{epoch_idx}/{self.config.epochs}]")
            self.logger.train_epoch_format(epoch_idx, train_result)
            if valid_result is not None:
                self.logger.valid_epoch_format(epoch_idx, valid_result)
                if valid_metric != "loss":
                    self.tensorboard_writer.add_scalar(f"eval/{valid_metric}", valid_result[valid_metric], epoch_idx)
            
            # scheduler step
            self.scheduler_step(valid_result)
            
            if load_best_model == True:
                if valid_metric == "loss":
                    current_valid_result = train_result["loss"]
                else:
                    current_valid_result = valid_result[valid_metric]
                
                if eval_compare(valid_metric, current_valid_result, best_result):
                    best_result = current_valid_result
                    best_epoch = epoch_idx
                    best_state_dict = deepcopy(self.model.state_dict())
                elif early_stop is not False and epoch_idx - best_epoch >= early_stop:
                    self.logger.info(f"early stop at epoch {epoch_idx}")
                    break
        
        if load_best_model == True:
            self.model.load_state_dict(best_state_dict)
            self.logger.info(f"load best model at epoch {best_epoch}")
        else:
            self.logger.info(f"load last model at epoch {epoch_idx}")

        # eval test 
        if test_loader is not None:
            test_result = self.evaluate(test_loader, **kwargs)
            self.logger.info(f"test_result : {test_result}")
            
        # save model
        if self.config.save_model != False:
            self.logger.info(f"save model at epoch {best_epoch} to {self.config.save_model}")
            if not os.path.exists(os.path.dirname(self.config.save_model)):
                os.makedirs(os.path.dirname(self.config.save_model))
            torch.save(self.model.state_dict(), self.config.save_model)     
            
        # close writer
        self.tensorboard_writer.close()

    def _train_epoch(self, 
                     train_loader,
                     **kwargs):
        # get loss function
        if "loss_fn" in kwargs:
            loss_fn = kwargs["loss_fn"]
        else:
            loss_fn = get_loss(config=self.config)
        if "pred_fn" in kwargs:
            pred_fn = kwargs["pred_fn"]
        else:
            pred_fn = self.pred_fn
        if "label_fn" in kwargs:
            label_fn = kwargs["label_fn"]
        else:
            label_fn = self.label_fn

        sum_loss = 0
        iter_data = (
            tqdm(
                train_loader,
                total=len(train_loader),
                ncols=100,
                desc = f"train "
            )
        )
        result_dict = None 
        for batch_idx, batched_data in enumerate(iter_data):
            self.model.train()
            self.optimizer.zero_grad()
            if pred_fn is not None: 
                batched_pred = pred_fn(batched_data)
            else:
                batched_pred = self.pred_fn(batched_data)
            if label_fn is not None:
                batched_label = label_fn(batched_data)
            else:
                batched_label = self.label_fn(batched_data)
            loss = loss_fn(batched_pred, batched_label)

            loss.backward()
            self.optimizer.step()
            sum_loss += loss.item() * batched_pred.shape[0]

            # self.model.eval()
            if "eval_fn" in kwargs:
                cur_result_dict = kwargs["eval_fn"](pred=batched_pred, label=batched_label)
            else:
                cur_result_dict = eval_all_metric(config=self.config, pred=batched_pred, label=batched_label)

            result_dict = eval_reduce([result_dict, cur_result_dict])
            iter_data.set_postfix(loss=loss.item(), **result_dict)
        result_dict["loss"] = sum_loss / result_dict["data_size"]
        
        return result_dict


class TaskTrainer(AbstractTrainer):
    def __init__(self, config, model):
        super(TaskTrainer, self).__init__(config=config, model=model)
    
    def pred_fn(self, batched_data):
        batched_pred = self.model(batched_data)
        return batched_pred

    def label_fn(self, batched_data):
        batched_label = batched_data["label"].to(self.device).float().squeeze(0)
        if len(batched_label.shape) == 0:
            batched_label = batched_label.unsqueeze(0)
        return batched_label
