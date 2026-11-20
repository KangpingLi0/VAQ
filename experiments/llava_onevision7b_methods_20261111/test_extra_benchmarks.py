import bootstrap
import unittest
from unittest.mock import patch

from lmms_eval import utils as task_utils
from lmms_eval.tasks.mmvp import utils as mmvp
from lmms_eval.tasks.naturalbench import utils as natural
from lmms_eval.tasks.mme_realworld import utils as realworld
from lmms_eval.tasks.mmt_bench_val import utils as mmt
from lmms_eval.tasks.mmbench.en_utils import mmbench_evaluator


class TestExtraBenchmarks(unittest.TestCase):
    def test_naturalbench_metric_keys_and_target_match_export(self):
        config=task_utils.load_yaml_config(str(bootstrap.SOURCE/'3rdparty/lmms-eval-new/lmms_eval/tasks/naturalbench/naturalbench.yaml'))
        self.assertEqual(config['doc_to_target'],'Answer')
        result=config['process_results']({'Index':0,'Question_Type':'yes_no'},['Yes'])
        self.assertEqual(set(result),{m['metric'] for m in config['metric_list']})

    def test_naturalbench_four_tuple_scores(self):
        rows=[{'id':i,'score':[1,0,0,1][i%4]} for i in range(7600)]
        for function in [natural.naturalbench_aggregate_results_G_ACC,
                         natural.naturalbench_aggregate_results_Q_ACC,
                         natural.naturalbench_aggregate_results_I_ACC,
                         natural.naturalbench_aggregate_results_ACC]:
            self.assertEqual(function(rows),1.0)
        rows[0]['score']=0
        self.assertAlmostEqual(natural.naturalbench_aggregate_results_G_ACC(rows),1899/1900)

    def test_mmvp_pair_accuracy_requires_both_answers(self):
        a=mmvp.mmvp_process_results({'Index':1,'Correct Answer':'(a)'},['A'])['mmvp_accuracy']
        b=mmvp.mmvp_process_results({'Index':2,'Correct Answer':'(b)'},['B'])['mmvp_accuracy']
        self.assertEqual(mmvp.mmvp_aggregate_pair_results([a,b]),1.0)
        b=mmvp.mmvp_process_results({'Index':2,'Correct Answer':'(b)'},['A'])['mmvp_accuracy']
        self.assertEqual(mmvp.mmvp_aggregate_pair_results([a,b]),0.0)
        self.assertEqual(mmvp.mmvp_aggregate_results([a,b]),0.5)

    def test_mmt_official_matching_and_overall_accuracy(self):
        doc={'index':'1','question':'Pick nine','answer':'I','hint':None,
            **{letter:str(i) for i,letter in enumerate('ABCDEFGHI',1)}}
        result=mmt.process_results(doc,['I'])['mmtbench_accuracy']
        self.assertTrue(result['correct'])
        self.assertIn('I. 9',mmt.doc_to_text(doc))
        wrong=mmt.process_results(doc,['A'])['mmtbench_accuracy']
        self.assertEqual(mmt.aggregate_results([result,wrong]),0.5)

    def test_mmbench_static_circular_scoring_never_calls_api(self):
        base={'index':1,'answer':'A','prediction':'A','A':'left','B':'right',
              'C':'up','D':'down','category':'test','L2-category':'test'}
        rotated={**base,'index':1000001,'answer':'B','prediction':'B'}
        with patch('requests.post',side_effect=AssertionError('Unexpected external judge call')):
            self.assertEqual(mmbench_evaluator.eval_result([base,rotated],eval_method='static')[0],1.0)
            rotated['prediction']='A'
            self.assertEqual(mmbench_evaluator.eval_result([base,rotated],eval_method='static')[0],0.0)

    def test_mme_realworld_native_accuracy_scale(self):
        doc={'index':1,'category':'perception/'+realworld.SUBTASKS[0],
             'l2-category':'test','answer':'A'}
        correct=realworld.mme_realworld_process_results(doc,['A'])['mme_realworld_score']
        wrong=realworld.mme_realworld_process_results({**doc,'index':2},['B'])['mme_realworld_score']
        self.assertEqual(realworld.mme_realworld_aggregate_results([correct,wrong]),0.5)


if __name__=='__main__':unittest.main()
