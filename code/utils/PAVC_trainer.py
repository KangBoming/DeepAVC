import torch
import numpy as np
from sklearn.metrics import accuracy_score ,precision_score,recall_score,roc_auc_score,average_precision_score,f1_score
import pandas as pd
from tqdm import tqdm

class PAVC_Trainer():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, 
                 device, model_name,  ddp=False, local_rank=0):
        self.args = args
        self.model_name = model_name
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.ddp = ddp
        self.local_rank = local_rank
            
    def _forward_epoch(self, model, batched_data):

        smiless, graphs, fps, mds, labels = batched_data


        fps = fps.to(self.device)
        mds = mds.to(self.device)
        graphs = graphs.to(self.device)
        labels = labels.to(self.device)

        predictions = model.forward_tune(graphs, fps, mds)
        return predictions, labels

    def train_epoch(self, model, train_loader):
        model.train()
        total_loss =  0
        for batch_idx, batched_data in tqdm(enumerate(train_loader), total=len(train_loader)):
            self.optimizer.zero_grad()

            predictions, labels = self._forward_epoch(model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)

            loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.lr_scheduler.step()
            total_loss += loss.item()

            if (batch_idx) % 600 == 0:
                print(f'[Batch {batch_idx+1}],{total_loss/ (batch_idx + 1)}')

    def fit(self, model, train_loader, val_loader):
        res_df_list = []
        for epoch in range(1, self.args['n_epochs'] + 1):
            self.train_epoch(model, train_loader)
            val_result = self.eval(model, val_loader)


            val_res_df = pd.DataFrame(val_result,index=[0])


            val_res_df['epoch'] = epoch 
            val_res_df['set'] = 'val'



            res_df_list.append(val_res_df)
            print(f"[Epoch{epoch}], val_auroc: {val_result['auroc']:.3f}, val_auprc: {val_result['auprc']:.3f} ")

        final_res_df = pd.concat(res_df_list, ignore_index=True)
        return  final_res_df

    def compute_metric(self, pred_logits, pred_label, true_label):
        acc= accuracy_score(y_true=true_label,y_pred=pred_label)
        recall = recall_score(y_true=true_label,y_pred=pred_label)
        prec = precision_score(y_true=true_label,y_pred=pred_label)
        f1 = f1_score(y_true=true_label,y_pred=pred_label)
        auroc = roc_auc_score(y_true=true_label,y_score=pred_logits)
        auprc=average_precision_score(y_true=true_label,y_score=pred_logits)
        metric_list =  [ 'acc','recall','prec','f1','auroc','auprc']
        result_list = [acc,recall,prec,f1,auroc,auprc]
        result_dict = dict(zip(metric_list,result_list))
        return result_dict 

    def eval(self, model, dataloader):
        model.eval()
        # predictions_all = []
        pred_labels_all = []
        pred_scores_all = []
        labels_all = []
        
        for batched_data in tqdm(dataloader, total=len(dataloader)):
            predictions, labels = self._forward_epoch(model, batched_data)
            predictions = predictions.squeeze(1)
            pred_scores = torch.sigmoid(predictions)
            pred_labels = torch.round(pred_scores)
            pred_scores_all.append(pred_scores.detach().cpu())
            pred_labels_all.append(pred_labels.detach().cpu())
            labels_all.append(labels.detach().cpu())
        
        all_labels = torch.cat(labels_all)
        all_pred_labels = torch.cat(pred_labels_all)
        all_pred_scores = torch.cat(pred_scores_all)

        result = self.compute_metric(all_pred_scores, all_pred_labels, all_labels)

        return result