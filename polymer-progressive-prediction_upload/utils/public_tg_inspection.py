import torch
import math
from torch import nn
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from models.public_tg_head import encode_digits,hard_digits
from utils.public_tg_data import save
from configs.public_tg_protocol import DIGIT_WEIGHTS,RECONSTRUCTION_COEFFICIENT,LOSS_TYPE

class EffectiveAudit(pl.Callback):
 def __init__(self,cfg,path):self.cfg=cfg;self.path=path
 def on_fit_start(self,trainer,model):
  h=model.head;names=['thousands','hundreds','tens']
  shapes={n:[(m.in_features,m.out_features) for m in getattr(h,n).modules() if isinstance(m,nn.Linear)] for n in names}
  assert shapes=={'thousands':[(300,1)],'hundreds':[(316, 64), (64, 1)],'tens':[(316,16),(16,1)]}
  assert not hasattr(h,'ones') and [sum(isinstance(m,nn.PReLU) for m in getattr(h,n).modules()) for n in names]==[0, 1, 1]
  assert sum(q.numel() for q in h.parameters())==26050
  ps=[m.p for m in h.modules() if isinstance(m,nn.Dropout)];assert ps==[.3]*3
  assert sum(q.numel() for q in h.embedding.parameters())==305 and not any(isinstance(m,nn.Embedding) for m in h.modules())
  opt=trainer.optimizers[0];assert type(opt) is torch.optim.AdamW and all(g['weight_decay']==0.0 for g in opt.param_groups)
  assert trainer.max_epochs==300 and [c.patience for c in trainer.callbacks if isinstance(c,EarlyStopping)]==[50]
  assert model.message_passing.output_dim==300 and model.message_passing.depth==3 and model.message_passing.dropout.p==0 and isinstance(model.bn,nn.BatchNorm1d)
  assert [model.init_lr,model.max_lr,model.final_lr,model.warmup_epochs]==[2e-3,2e-3,1e-7,0]
  lrconfig=trainer.lr_scheduler_configs[0];schedule=lrconfig.scheduler
  assert type(schedule) is torch.optim.lr_scheduler.CosineAnnealingLR and schedule.T_max==300 and schedule.eta_min==1e-7
  assert lrconfig.interval=='epoch' and lrconfig.frequency==1
  assert opt.defaults['betas']==(.9,.999) and opt.defaults['eps']==1e-8
  assert set(id(p) for g in opt.param_groups for p in g['params'])==set(id(p) for p in model.parameters() if p.requires_grad)
  self.count={True:0,False:0};self.backward=False
  self.report={'head_linear_shapes':shapes,'actual_head_parameters':26050,'prelu_counts':[0, 1, 1],'actual_head_dropouts':ps,'config':self.cfg,'target_unit':'log10(K)','target_source':'harmonized_csv_logTg','target_scale':1000,'fixed_integer_part':None,'thousands_condition_source':'molecular_features_only','condition_encoder_type':'normalized_scalar_mlp','condition_encoder_parameters':305,'teacher_forcing':True,'training_condition_source':'true_truncated_log1000_digits','inference_condition_source':'predicted_half_up_digits','loss_type':LOSS_TYPE,'digit_weights':list(DIGIT_WEIGHTS),'reconstruction_loss_coefficient':RECONSTRUCTION_COEFFICIENT,'reconstruction_forward':None,'reconstruction_backward':None,'ste_applied_to_auxiliary_only':False,'auxiliary_reconstruction_removed':True,'continuous_reconstruction_loss_coefficient':0,'evaluation_reconstruction':'half_up_top3_digits_divide1000','selection_metric':'val_rmse_log10','max_epochs':300,'patience':50,'n_predicted_digits':3,'predicted_places':[1000,100,10],'log10_resolution':.01,'batch_size':64,'optimizer':'AdamW','optimizer_betas':[.9,.999],'optimizer_eps':1e-8,'scheduler':'CosineAnnealingLR','scheduler_interval':'epoch','scheduler_frequency':1,'scheduler_t_max':300,'initial_lr':2e-3,'eta_min':1e-7,'warmup_epochs':0,'warm_restarts':False,'optimizer_scheduler_verified':True,'learning_rate_history':[],'actual_optimizer_weight_decays':[g['weight_decay'] for g in opt.param_groups]}
  def start(module,args):
   target=args[1] if len(args)>1 else None;self.training=module.training
   if self.training:assert target is not None;self.true_digits=encode_digits(target)[1]
   else:assert target is None
  def output(module,args,raw):
   if not self.training:self.expected=hard_digits(9*torch.sigmoid(raw).squeeze(-1))
  def condition(module,args):
   if self.count[self.training]<2:
    d=args[0];assert d.dtype==torch.long
    expected=self.true_digits[:,self.count[True]] if self.training else self.expected
    assert torch.equal(d,expected);self.count[self.training]+=1
  def hundred_input(module,args):assert args[0].shape[1]==300
  self.hooks=[h.thousands.register_forward_pre_hook(hundred_input),h.register_forward_pre_hook(start),h.embedding.register_forward_pre_hook(condition)]+[getattr(h,n).register_forward_hook(output) for n in names[:2]]
  save(self.path,self.report)
 def on_train_epoch_start(self,trainer,model):
  epoch=int(model.current_epoch);lr=trainer.optimizers[0].param_groups[0]['lr']
  expected=1e-7+.5*(2e-3-1e-7)*(1+math.cos(math.pi*epoch/300))
  assert abs(lr-expected)<1e-12 and trainer.lr_scheduler_configs[0].scheduler.last_epoch==epoch
  self.report['learning_rate_history'].append({'epoch_index':epoch,'lr':lr})
 def on_train_batch_start(self,trainer,model,batch,batch_idx):
  lr=trainer.optimizers[0].param_groups[0]['lr']
  assert lr==self.report['learning_rate_history'][-1]['lr']
  model.log('train_learning_rate',lr,on_step=False,on_epoch=True,batch_size=int(batch[3].shape[0]))
  if 'observed_target_min' not in self.report:
   y=batch[3];assert torch.isfinite(y).all() and y.min()>=2 and y.max()<3
   self.report.update(observed_target_min=float(y.min()),observed_target_max=float(y.max()),observed_batch_size=int(y.shape[0]),log10_training_targets_verified=True)
 def on_after_backward(self,trainer,model):
  if not self.backward:
   assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
   ex=model.loss_audit;assert tuple(DIGIT_WEIGHTS)==(100,80,20) and RECONSTRUCTION_COEFFICIENT==0 and LOSS_TYPE=='weighted_digit_mse'
   assert encode_digits(torch.tensor(ex['targets']))[1].tolist()==ex['digit_targets']
   assert ex['auxiliary_reconstruction_removed'] and ex['reconstruction_loss_coefficient']==0 and 'reconstruction_loss' not in ex
   digit=sum(sum(w*(v-t)**2 for w,v,t in zip(DIGIT_WEIGHTS,row,truth)) for row,truth in zip(ex['raw_digits'],ex['digit_targets']))/len(ex['targets'])
   assert abs(digit-ex['digit_loss'])<max(1e-4,1e-6*abs(digit))
   assert ex['digit_loss']==ex['loss']
   self.report.update(loss_example=ex,weighted_digit_loss_verified=True,digit_only_loss_verified=True,auxiliary_reconstruction_removed=True);self.backward=True
 def on_fit_end(self,trainer,model):
  for h in self.hooks:h.remove()
  assert self.count[True]==self.count[False]==2 and self.backward
  self.report.update(status='passed',true_training_condition_verified=True,predicted_inference_condition_verified=True,thousands_molecular_features_only_verified=True,finite_gradients_verified=True);save(self.path,self.report)
