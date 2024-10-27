from stemflow.utils.lazyloading import LazyLoadingEnsembleDict
from stemflow.model.dummy_model import dummy_model1

def test_LazyLoadingEnsembleDict_functions():
    dd = LazyLoadingEnsembleDict('./test_LazyLoadingEnsembleDict')
    model_dict = {'0_0_2_2_model', '0_2_2_2_model', '0_3_2_2_model', '0_4_2_2_model',
                  '1_0_2_2_model', '1_2_2_2_model', '1_3_2_2_model', '1_4_2_2_model',
                  '2_0_2_2_model', '2_2_2_2_model', '2_3_2_2_model', '2_4_2_2_model',
                  '3_0_2_2_model', '3_2_2_2_model', '3_3_2_2_model', '3_4_2_2_model'}
    
    for model_name in model_dict:
        dd[model_name] = dummy_model1(1)
        
    assert len(list(dd.items())) > 0
    assert len(list(dd.keys())) > 0
    assert len(dd)>0
    assert len(list(dd.keys()))==len(list(dd.values()))
    iter(dd)
    assert '0_0_2_2_model' in dd
    dd.get('0_0_2_2_model')
    a = dd.pop('0_0_2_2_model')
    assert a != None
    dd.update({'0_9_2_2_model':dummy_model1(1)})
    cc = dd.copy()
    assert len(cc) == len(dd)
    cc.dump_ensemble(0)
    cc.load_model('0_9_2_2_model')
    cc.dump_ensemble(1)
    cc.load_ensemble(1)
    cc.dump_ensemble(1)
    cc.delete_ensemble(1)
    assert '1' not in cc.ensemble_models
    del cc['3_0_2_2_model']
    assert '3_0_2_2_model' not in cc
    cc.clear()
    assert len(cc)==0
