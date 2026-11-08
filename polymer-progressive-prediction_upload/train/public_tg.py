"""Ten-fold/ten-seed packaging of the final model; test all validation-best checkpoints after training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping,ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from utils.public_tg_data import ROOT,BASE,save,load_development,load_test,make_loader,metrics,predict_validation
from models.public_tg import ProgressiveTgModel
from utils.public_tg_inspection import EffectiveAudit
from configs.public_tg import CONFIG,SEEDS,N_FOLDS,identity
from configs.public_tg_protocol import MAX_EPOCHS,PATIENCE,DIGIT_WEIGHTS,LOSS_TYPE,RECONSTRUCTION_COEFFICIENT


def location(fold,seed):
    return ROOT/'results'/f'fold_{fold:02d}'/f'seed_{seed}'


def train(task):
    fold,seed=identity(task);out=location(fold,seed);out.mkdir(parents=True,exist_ok=True)
    (out/'training_claim').mkdir()
    if (out/'completed.json').exists():
        raise FileExistsError('Refusing duplicate completed training')
    frame,splits=load_development();assert len(splits)==N_FOLDS==10
    a,b=splits[str(fold)]['train'],splits[str(fold)]['validation']
    pl.seed_everything(seed,workers=True)
    model=ProgressiveTgModel(**CONFIG)
    callback=ModelCheckpoint(dirpath=out/'checkpoints',filename='best',monitor='val_rmse_log10',mode='min',save_top_k=1)
    trainer=pl.Trainer(accelerator='gpu',devices=1,max_epochs=MAX_EPOCHS,deterministic=True,
                        enable_progress_bar=False,logger=CSVLogger(out,name='training'),
                        callbacks=[callback,EarlyStopping(monitor='val_rmse_log10',mode='min',patience=PATIENCE),
                                   EffectiveAudit(CONFIG,out/'effective_config.json')])
    trainer.fit(model,make_loader(frame.iloc[a],64,shuffle=True),make_loader(frame.iloc[b],64))
    checkpoint=Path(callback.best_model_path)
    best=ProgressiveTgModel.load_from_checkpoint(checkpoint,map_location='cpu')
    assert best.weight_decay==CONFIG['weight_decay']
    assert [m.p for m in best.head.modules() if isinstance(m,torch.nn.Dropout)]==[CONFIG['dropout']]*3
    val=predict_validation(best,frame.iloc[b],trainer,out/'validation_predictions.csv')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    save(out/'completed.json',{'architecture_variant':'A_tens16','status':'completed','task':task,'fold':fold,'seed':seed,'n_folds':10,
          'config':CONFIG,'n_train':len(a),'n_validation':len(b),'max_epochs':MAX_EPOCHS,
          'patience':PATIENCE,'optimizer':'AdamW','scheduler':'CosineAnnealingLR','scheduler_interval':'epoch','scheduler_t_max':300,'initial_lr':0.002,'eta_min':1e-07,'warmup_epochs':0,'warm_restarts':False,'selected_checkpoint_scheduler_state':state['lr_schedulers'][0],'selected_checkpoint_optimizer_lrs':[g['lr'] for g in state['optimizer_states'][0]['param_groups']],'digit_weights':list(DIGIT_WEIGHTS),'loss_type':LOSS_TYPE,'reconstruction_loss_coefficient':RECONSTRUCTION_COEFFICIENT,'target_unit':'log10(K)','target_source':'harmonized_csv_logTg','target_scale':1000,'fixed_integer_part':None,'teacher_forcing':True,'checkpoint':str(checkpoint.resolve()),
          'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
          'best_epoch':int(state['epoch'])+1,'stop_epoch':int(trainer.current_epoch),'validation_log10':val})


def finalize():
    frame,splits=load_development();records=[]
    submission=json.loads((ROOT/'provenance/public_tg/package_manifest.json').read_text())
    for name,h in submission['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
    for task in range(N_FOLDS*len(SEEDS)):
        fold,seed=identity(task);path=location(fold,seed)/'completed.json'
        r=json.loads(path.read_text())
        assert r['status']=='completed' and r['task']==task and r['fold']==fold and r['seed']==seed
        assert r['config']==CONFIG and r['n_folds']==10 and r['max_epochs']==300 and r['patience']==50
        assert r['digit_weights']==list(DIGIT_WEIGHTS) and r['loss_type']==LOSS_TYPE and r['reconstruction_loss_coefficient']==0. and r['target_unit']=='log10(K)' and r['target_scale']==1000 and r['fixed_integer_part'] is None
        assert hashlib.sha256(Path(r['checkpoint']).read_bytes()).hexdigest()==r['checkpoint_sha256']
        effective=json.loads((path.parent/'effective_config.json').read_text())
        assert effective['status']=='passed' and effective['actual_head_dropouts']==[.3]*3
        assert effective['head_linear_shapes']=={'thousands':[[300,1]],'hundreds':[[316, 64], [64, 1]],'tens':[[316,16],[16,1]]}
        assert all(w==0.0 for w in effective['actual_optimizer_weight_decays'])
        assert effective['optimizer']=='AdamW' and effective['scheduler']=='CosineAnnealingLR' and effective['optimizer_scheduler_verified']
        assert effective['scheduler_t_max']==300 and effective['eta_min']==1e-7 and effective['initial_lr']==2e-3 and effective['warmup_epochs']==0
        assert r['optimizer']=='AdamW' and r['scheduler']=='CosineAnnealingLR'
        assert r['selected_checkpoint_scheduler_state']['T_max']==300 and r['selected_checkpoint_scheduler_state']['eta_min']==1e-7
        assert effective['digit_only_loss_verified'] and effective['auxiliary_reconstruction_removed'] and effective['log10_training_targets_verified']
        assert effective['true_training_condition_verified'] and effective['predicted_inference_condition_verified'] and effective['reconstruction_loss_coefficient']==0.
        records.append(r)
    assert len(records)==100 and len({r['checkpoint_sha256'] for r in records})==100
    baseline=json.loads((ROOT/'provenance/public_tg/data_manifest.json').read_text())
    tpath=BASE/'data/public_tg/test_data.csv'
    tsha=hashlib.sha256(tpath.read_bytes()).hexdigest()
    assert tsha==baseline['data_sha256']['test_data.csv']
    save(ROOT/'provenance/public_tg/frozen_test_manifest.json',{'n_models':100,'n_folds':10,'seeds':SEEDS,
         'config':CONFIG,'selection':'best validation log10 RMSE per fold and seed',
         'test_sha256':tsha,'models':records,'source_sha256':submission['source_sha256'],'data_sha256':baseline['data_sha256'],'split_sha256':hashlib.sha256((ROOT/'provenance/public_tg/folds.json').read_bytes()).hexdigest()})
    test=load_test();predictions={};oof={seed:np.full(len(frame),np.nan) for seed in SEEDS};scores={seed:[] for seed in SEEDS}
    trainer=pl.Trainer(accelerator='gpu',devices=1,logger=False,enable_checkpointing=False,enable_progress_bar=False)
    for record in records:
        fold,seed=record['fold'],record['seed'];source=location(fold,seed)
        b=splits[str(fold)]['validation'];val=pd.read_csv(source/'validation_predictions.csv')
        assert val.row_id.tolist()==b
        score=metrics(frame.iloc[b].logTg,val.pred_logTg.values)
        for name in score:assert abs(score[name]-record['validation_log10'][name])<1e-5
        scores[seed].append(score);oof[seed][b]=val.pred_logTg.values
        path=Path(record['checkpoint']);assert hashlib.sha256(path.read_bytes()).hexdigest()==record['checkpoint_sha256']
        model=ProgressiveTgModel.load_from_checkpoint(path,map_location='cpu')
        out=ROOT/'final_test'/f'seed_{seed}'/f'fold_{fold:02d}';out.mkdir(parents=True,exist_ok=True)
        print(f'Testing tenfold model seed={seed} fold={fold}',flush=True)
        score=predict_validation(model,test,trainer,out/'test_predictions.csv')
        save(out/'completed.json',{'status':'completed','seed':seed,'fold':fold,'n_test':209,
             'config':CONFIG,'checkpoint_sha256':record['checkpoint_sha256'],'test_log10':score})
        predictions[seed,fold]=pd.read_csv(out/'test_predictions.csv').pred_logTg.values
        del model
        torch.cuda.empty_cache()
    dest=ROOT/'aggregate';dest.mkdir(exist_ok=True);per_seed=[];seed_validation=[]
    for seed in SEEDS:
        assert np.isfinite(oof[seed]).all()
        p=np.mean([predictions[seed,fold] for fold in range(N_FOLDS)],axis=0)
        m=metrics(test.logTg,p);per_seed.append({'seed':seed,'n_models':10,'test_log10':m})
        pd.DataFrame({'row_id':test.index,'true_logTg':test.logTg,'pred_logTg':p}).to_csv(dest/f'test_predictions_seed_{seed}.csv',index=False)
        row={'seed':seed}
        for name in ['mae','rmse','r2']:
            row['mean_'+name]=float(np.mean([m[name] for m in scores[seed]]))
            row['fold_sd_'+name]=float(np.std([m[name] for m in scores[seed]],ddof=0))
        seed_validation.append(row)
    p=np.mean(list(predictions.values()),axis=0)
    pd.DataFrame({'row_id':test.index,'true_logTg':test.logTg,'pred_logTg':p}).to_csv(dest/'test_predictions.csv',index=False)
    mean_oof=np.mean([oof[seed] for seed in SEEDS],axis=0)
    pd.DataFrame({'row_id':frame.index,'true_logTg':frame.logTg,'pred_logTg':mean_oof}).to_csv(dest/'validation_oof_predictions.csv',index=False)
    ensemble_folds=[metrics(frame.iloc[splits[str(f)]['validation']].logTg,mean_oof[splits[str(f)]['validation']]) for f in range(N_FOLDS)]
    validation={name:{'mean':float(np.mean([r['mean_'+name] for r in seed_validation])),
                      'seed_population_sd':float(np.std([r['mean_'+name] for r in seed_validation],ddof=0)),
                      'ensemble_fold_mean':float(np.mean([r[name] for r in ensemble_folds]))} for name in ['mae','rmse','r2']}
    summary={'n_models':100,'n_folds':10,'n_seeds':10,'n_test':209,'config':CONFIG,
          'selection':'Log10 top-three-digit weighted digit MSE only; no reconstruction loss; checkpoint chosen by hard autoregressive validation log10 RMSE',
          'architecture':{'thousands':[300,1],'hundreds':[316, 64, 1],'tens':[316,16,1],'prelu_counts':[0, 1, 1]},
          'protocol':{'digit_weights':list(DIGIT_WEIGHTS),'max_epochs':MAX_EPOCHS,'patience':PATIENCE,
                      'batch_size':64,'condition_output_dim':16,'max_lr':2e-3,'reconstruction_loss_coefficient':0.,'reconstruction_forward':None,'reconstruction_backward':None},
          'optimizer':'AdamW','scheduler':'CosineAnnealingLR','scheduler_interval':'epoch','scheduler_t_max':300,'initial_lr':0.002,'eta_min':1e-07,'warmup_epochs':0,'warm_restarts':False,'loss_type':LOSS_TYPE,'auxiliary_reconstruction_removed':True,'digit_weights':list(DIGIT_WEIGHTS),'reconstruction_loss_coefficient':0.,'reconstruction_forward':None,'reconstruction_backward':None,'target_unit':'log10(K)','target_source':'harmonized_csv_logTg','fixed_integer_part':None,'teacher_forcing':True,'condition_encoder_type':'normalized_scalar_mlp','condition_encoder_parameters':305,'test_log10':metrics(test.logTg,p),'per_seed':per_seed,
          'seed_statistics':{name:{'mean':float(np.mean([r['test_log10'][name] for r in per_seed])),
                                    'population_sd':float(np.std([r['test_log10'][name] for r in per_seed],ddof=0))} for name in ['mae','rmse','r2']},
          'validation':validation,'validation_oof_log10':metrics(frame.logTg,mean_oof),
          'training_records':records,'n_predicted_digits':3,'predicted_places':[1000,100,10],'log10_resolution':.01,'test_input_overlap_rows':8,'dataset_version':'median_harmonized_le10K_v1_test_exact_input_label_excluded_v1'}
    pd.DataFrame([{'seed':r['seed'],**r['test_log10']} for r in per_seed]).to_csv(dest/'per_seed_test_results.csv',index=False)
    pd.DataFrame(seed_validation).to_csv(dest/'per_seed_validation_results.csv',index=False)
    save(dest/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='training_records'},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['train','finalize']);p.add_argument('--task',type=int,choices=range(100));a=p.parse_args()
    if a.command=='train':
        if a.task is None:p.error('--task required')
        train(a.task)
    else:finalize()
