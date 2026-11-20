from mobile_health.experiment import dataset,evaluate
def test_shape():
 x,y=dataset(20,2);assert x.shape==(20,9) and len(y)==20
def test_metrics():assert all(0<=v<=1 for m in evaluate(3).values() for v in m.values())
