"""Frozen K=2 training study, queued after the existing single-view experiments."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_synthesis_history_study import normalized, comparison_protocol, digest, save, now, worker, check_sources

COMMON = Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],cwd=ROOT,text=True).strip())
MAIN_ROOT = (ROOT/COMMON).resolve().parent
PRIOR = MAIN_ROOT/'analysis_scripts/synthesis_history_compact_study_20260913'
STUDY = ROOT/'analysis_scripts/synthesis_multiview_study_20260913'


def prepare():
    old = json.loads((PRIOR/'plan.json').read_text())
    controls = {k:v for k,v in old['controls'].items() if k in ('none_43','none_44','none_45','baseline_43')}
    jobs = []
    for seed in (43,44,45):
        arguments = ['--models','clip_adapter','--datasets','cifar100','--methods','fedavg',
            '--defenses','risk_synthesis','--attacks','all','--seeds',str(seed),'--rounds','100',
            '--set',f"confirmation_split_manifest={old['confirmation_manifest']}",
            '--set',f"confirmation_split_sha256={old['confirmation_manifest_sha256']}",
            '--set','defense.synthesis.views_per_record=2',
            '--set','defense.synthesis.risk_history=none',
            '--set','defense.synthesis.candidate_selection=first_semantic',
            '--set','defense.synthesis.mode=risk']
        config = normalized(arguments)
        protocol = comparison_protocol(config)
        if protocol != controls[f'none_{seed}']['protocol']:
            raise ValueError('Multiview protocol does not match the original-record control.')
        jobs.append(dict(id=f'multiview_{seed}',arm='multiview',seed=seed,arguments=arguments,
                         protocol=protocol,defense=config['defense']))
    sources = set(old['source_hashes']) | {'scripts/run_synthesis_multiview_study.py',
        'scripts/run_synthesis_multiview.py','scripts/verify_synthesis_multiview.py',
        'scripts/analyze_risk_synthesis.py','scripts/paired_synthesis_uncertainty.py',
        'scripts/confirmation_duplicate_sensitivity.py','scripts/summarize_synthesis_exposure_distribution.py',
        'scripts/analyze_synthesis_history_study.py'}
    plan = dict(schema_version=1, prepared_at_utc=now(),jobs=jobs,controls=controls,
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes={p:digest(ROOT/p) for p in sorted(sources)},
        prior_study=str(PRIOR),prior_plan_sha256=digest(PRIOR/'plan.json'),
        pending_baseline_controls={f'baseline_{s}':str(PRIOR/f'baseline_{s}.json') for s in (44,45)},
        confirmation_manifest=old['confirmation_manifest'],confirmation_manifest_sha256=old['confirmation_manifest_sha256'],
        environment=old['environment'],
        hypothesis='Two independently generated virtual views both contribute to each original-record update.',
        scope='CLIP transformer Adapter/CIFAR100; K=2; seeds43/44/45; 100 FedAvg rounds; 100 originals/class; '
              '10 IID clients; target0; 1000 original members and 1000 independent evaluation nonmembers. '
              'Existing studied source/seed identities; not untouched-data or cross-model confirmation.',
        training='Mean CE across K views and B originals; one optimizer step per original batch; '
                 'same risk per original across views; semantic retry budget is separate from trained views.',
        hyperparameters=dict(views_per_record=2,search_performed=False,risk_history='none',candidate_selection='first_semantic',
                             unchanged_generation_parameters=config['defense']['synthesis'],
                             caveat='K is an explicit view-count choice. No new loss coefficient, threshold or risk-history parameter.'),
        criteria=dict(overall_vs_none='Each seed: max11 AUC drop >=0.02, max11 TPR@1% drop >0, accuracy loss <=0.02.',
                      added_value_vs_single_view='Each seed: max11 AUC and TPR both lower; accuracy loss <=0.02.',
                      reporting='All scheduled seeds and 100-round checkpoints; all11 attacks; paired candidate intervals, '
                                'class-conditional metrics, original/view counts, semantic failures and runtime. No post-hoc K tuning.'))
    STUDY.mkdir(exist_ok=False)
    save(STUDY/'plan.json',plan,exclusive=True)
    print(json.dumps(dict(plan=str(STUDY/'plan.json'),sha256=digest(STUDY/'plan.json'),new_jobs=3),indent=2))


def prior_workers_alive():
    live = []
    for path in PRIOR.glob('worker_gpu*.json'):
        pid = int(json.loads(path.read_text())['pid'])
        try:
            command = Path(f'/proc/{pid}/cmdline').read_bytes()
        except FileNotFoundError:
            continue
        if b'run_synthesis_history_study.py' in command and Path(f'/proc/{pid}/cwd').resolve()==MAIN_ROOT:
            live.append(pid)
    return live


def run(gpu):
    plan = json.loads((STUDY/'plan.json').read_text())
    check_sources(plan)
    gate = STUDY/f'dependency_gpu{gpu}.json'
    state = dict(status='waiting_for_prior_workers',pid=os.getpid(),gpu=gpu,started_at_utc=now())
    save(gate,state,exclusive=True)
    try:
        while True:
            if digest(PRIOR/'plan.json') != plan['prior_plan_sha256']:
                raise ValueError('Prior frozen plan changed.')
            old = json.loads((PRIOR/'plan.json').read_text())
            states = [json.loads((PRIOR/f"{j['id']}.json").read_text()) if (PRIOR/f"{j['id']}.json").exists()
                      else {'status':'unclaimed'} for j in old['jobs']]
            if any(s['status']=='failed' for s in states):
                raise RuntimeError('Prior study failed; preserve results and inspect the failure before scheduling.')
            live = prior_workers_alive()
            done = all(s['status']=='completed' for s in states)
            if done and not live:
                break
            if not done and not live:
                raise RuntimeError('Prior workers are absent with unfinished tasks; not restarting them.')
            state.update(verified_live_prior_workers=live,prior_completed=sum(s['status']=='completed' for s in states))
            save(gate,state)
            time.sleep(30)
        state.update(status='starting_multiview_worker')
        save(gate,state)
        worker(STUDY,gpu,True)
        state.update(status='finished_owned_queue')
        save(gate,state)
    except BaseException as error:
        state.update(status='failed',error=repr(error))
        save(gate,state)
        raise


def analyze():
    from scripts.analyze_synthesis_history_study import metrics, compare
    from scripts.analyze_risk_synthesis import analyze as analyze_runs
    from scripts.paired_synthesis_uncertainty import run as resample
    from scripts.summarize_synthesis_exposure_distribution import run as exposures
    from scripts.confirmation_duplicate_sensitivity import run as duplicates
    from scripts.verify_synthesis_history import verify
    plan = json.loads((STUDY/'plan.json').read_text())
    check_sources(plan)
    paths = {k:Path(v['directory']) for k,v in plan['controls'].items()}
    for key,path in plan['pending_baseline_controls'].items():
        state = json.loads(Path(path).read_text())
        if state['status']!='completed' or state['returncode']!=0 or state['plan_sha256']!=plan['prior_plan_sha256']:
            raise ValueError('Unverified single-view baseline.')
        paths[key] = Path(state['result_directories'][0])
    mechanisms = {}
    for job in plan['jobs']:
        state = json.loads((STUDY/f"{job['id']}.json").read_text())
        if state['status']!='completed' or state['returncode']!=0 or state['plan_sha256']!=digest(STUDY/'plan.json'):
            raise ValueError('Uncompleted multiview run.')
        paths[job['id']] = Path(state['result_directories'][0])
        mechanisms[job['id']] = verify(paths[job['id']]/'risk_synthesis')
    output = STUDY/'analysis'
    output.mkdir(exist_ok=False)
    verified = analyze_runs(list(paths.values()),output/'metrics')
    by_path = {Path(r['path']).resolve():r for r in verified['runs']}
    records = {k:by_path[p.resolve()] for k,p in paths.items()}
    if any(not r['complete'] or len(r['attacks'])!=11 for r in records.values()):
        raise ValueError('Incomplete formal attack results.')
    comparisons = [compare(f'multiview_{s}',f'{arm}_{s}',records) for s in (43,44,45) for arm in ('none','baseline')]
    for row in comparisons:
        resample(output/'metrics/verified_results.json',output/f"{row['treatment']}_vs_{row['control']}",2000,20260913,
                 treatment=records[row['treatment']]['run'],control_name=records[row['control']]['run'])
    exposures(output/'metrics/verified_results.json',output/'exposures')
    duplicates(output/'metrics/verified_results.json',
               MAIN_ROOT/'analysis_scripts/risk_synthesis_confirmation_exact_image_identity_20260912.json',output/'duplicates')
    outcome = dict(status='completed',scope=plan['scope'],plan_sha256=digest(STUDY/'plan.json'),
                   arms=[metrics(k,r) for k,r in records.items()],comparisons=comparisons,mechanisms=mechanisms)
    save(output/'outcome.json',outcome,exclusive=True)
    print(json.dumps({'status':'completed','output':str(output)},indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','run','analyze'))
    parser.add_argument('--gpu',type=int,choices=(0,1),default=0)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action=='prepare': prepare()
    elif args.action=='run': run(args.gpu)
    else: analyze()
