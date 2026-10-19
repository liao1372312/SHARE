"""Run frozen SHARE-V2 once on the sealed V2 confirmation holdout."""
from __future__ import annotations
import hashlib,json,math,sys
from pathlib import Path
from statistics import mean,stdev
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_share_online_hard_cap import OnlineHardHumanCapPolicy
from share.data import load_episodes
from share.dqn import DQNPolicy
from share.evaluation import evaluate_policy
from share.features import FeatureEncoder
from share.types import AuthorityAction,RewardConfig
DATA=ROOT/'datasets/share_v2_confirmation_holdout_20260827/episodes.jsonl'
ENC=ROOT/'results/share_v2_stop_penalty_tuning_20260827/p060/feature_encoder.json'
CK={17:ROOT/'results/share_v2_stop_penalty_tuning_20260827/p060/share_v2_seed_17.pt',**{s:ROOT/f'results/share_v2_stop_penalty_tuning_20260827/p060_remaining/share_v2_seed_{s}.pt' for s in(29,43,71,101)}}
def h(p):
 d=hashlib.sha256();d.update(p.read_bytes());return d.hexdigest()
def main():
 out=ROOT/'results/share_v2_confirmation_locked_20260827';out.mkdir(parents=True,exist_ok=True)
 if (out/'FINAL_RESULTS.json').exists():raise SystemExit('confirmation already executed')
 eps=load_episodes(DATA);cap=math.floor(.2*len(eps));manifest={'protocol':'share-v2-confirmation-once-v1','dataset_sha256':h(DATA),'episodes':len(eps),'cap':cap,'encoder_sha256':h(ENC),'checkpoints':{str(s):h(p) for s,p in CK.items()},'evaluator_sha256':h(ROOT/'scripts/run_share_online_hard_cap.py'),'status':'sealed_before_execution'}
 (out/'pre_evaluation_manifest.json').write_text(json.dumps(manifest,indent=2))
 enc=FeatureEncoder.load(ENC);train=RewardConfig(rollback_penalty=.60);ev=RewardConfig();rows=[]
 for s,p in CK.items():
  pol=DQNPolicy.load(str(p),enc,train,require_reward_config=True);pol.name='share_v2';wrap=OnlineHardHumanCapPolicy(pol,cap,False);raw=evaluate_policy(wrap,eps,ev,.2,None);r={k:v for k,v in raw.items() if k!='records'};r.update(seed=s,hard_human_cap=cap,human_interventions=wrap.human_interventions,cap_respected=wrap.human_interventions<=cap);rows.append(r)
 metrics=('repair_sr','hir','uar','stopped_rate','normalized_return','collaboration_regret');agg={m+'_mean':mean(float(r[m]) for r in rows) for m in metrics}|{m+'_std':stdev(float(r[m]) for r in rows) for m in metrics}|{'all_caps_respected':all(r['cap_respected'] for r in rows)}
 result={'evidence_class':'new_workflow_group_disjoint_replayable_synthetic_confirmation','manifest_sha256':h(out/'pre_evaluation_manifest.json'),'per_seed':rows,'aggregate':agg};(out/'FINAL_RESULTS.json').write_text(json.dumps(result,indent=2));print(json.dumps(agg,indent=2))
if __name__=='__main__':main()
