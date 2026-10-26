"""Independent finite-population identities and information-boundary regressions."""
import hashlib, tempfile, unittest
from pathlib import Path
import numpy as np
import pandas as pd
from criteo_release.core import (FEATURES,COLUMNS,features,validate_frame,fingerprint,mix64,split_ids,
    assert_group_disjoint,budget_weights,augmented_contrast,clustered_summary,original_response,
    selected_response,integrated_contribution)
from criteo_release.data import verify_source,verify_protocol
from criteo_release.evaluation import select_ranking
from criteo_release.models import outcome_model

class InputAndIsolationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(22)
        self.frame=pd.DataFrame(rng.normal(size=(60,12)),columns=FEATURES)
        for k in COLUMNS[12:]:self.frame[k]=rng.integers(0,2,60)
    def test_exact_source_schema(self):
        validate_frame(self.frame)
        with self.assertRaises(ValueError):validate_frame(self.frame[COLUMNS[::-1]])
    def test_no_exposure_or_other_label_feature_leakage(self):
        before=features(self.frame)
        for k in ["treatment","conversion","visit","exposure"]:self.frame[k]=999.
        np.testing.assert_array_equal(before,features(self.frame))
    def test_no_fractional_label_truncation(self):
        for col in COLUMNS[12:]:
            bad=self.frame.astype(float);bad.loc[0,col]=.2
            with self.assertRaises(ValueError):validate_frame(bad)
    def test_nonfinite_source_fails(self):
        for value in [np.nan,np.inf,-np.inf]:
            bad=self.frame.copy();bad.loc[0,"f0"]=value
            with self.assertRaises(ValueError):validate_frame(bad)
    def test_fingerprint_is_independent_of_labels_and_order(self):
        fp=fingerprint(features(self.frame))
        changed=self.frame.copy()
        changed.loc[:,COLUMNS[12:]]=1-changed[COLUMNS[12:]]
        np.testing.assert_array_equal(fp,fingerprint(features(changed)))
        perm=np.random.default_rng(7).permutation(len(fp))
        np.testing.assert_array_equal(fp[perm],fingerprint(features(changed.iloc[perm])))
    def test_identical_vectors_and_signed_zero_group(self):
        x=np.ones((3,12));x[0,0]=-0.;x[1,0]=0.;x[2,0]=1.
        h=fingerprint(x)
        self.assertEqual(h[0],h[1]);self.assertNotEqual(h[1],h[2])
        np.testing.assert_array_equal(split_ids(h)[:2],[split_ids(h)[0]]*2)
    def test_group_cross_split_and_crossfit_rejected(self):
        fp=np.array([1,1,2,2,3,3],dtype=np.uint64)
        assert_group_disjoint(fp,split_ids(fp))
        assert_group_disjoint(fp,mix64(fp,2026102603)%3)
        with self.assertRaises(ValueError):assert_group_disjoint(fp,np.arange(6))
    def test_source_hash_and_size_guards(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"source.gz";p.write_bytes(b"abc")
            self.assertEqual(verify_source(p,hashlib.sha256(b"abc").hexdigest(),3),
                             hashlib.sha256(b"abc").hexdigest())
            with self.assertRaises(ValueError):verify_source(p,"0"*64,3)
            with self.assertRaises(ValueError):verify_source(p,hashlib.sha256(b"abc").hexdigest(),4)
    def test_frozen_protocol_hash(self):verify_protocol()

class StatisticalOracleTests(unittest.TestCase):
    def test_augmented_identity_correct_assignment_wrong_regression(self):
        p0,p1,e=.1,.25,.7
        weights=np.array([(1-e)*(1-p0),(1-e)*p0,e*(1-p1),e*p1])
        t=np.array([0,0,1,1]);y=np.array([0,1,0,1])
        psi=augmented_contrast(y,t,np.full(4,.8),np.full(4,.02),np.full(4,e))
        self.assertAlmostEqual(float(weights@psi),p1-p0,14)
    def test_augmented_identity_correct_regression_wrong_assignment(self):
        p0,p1,e=.1,.25,.7
        weights=np.array([(1-e)*(1-p0),(1-e)*p0,e*(1-p1),e*p1])
        psi=augmented_contrast([0,1,0,1],[0,0,1,1],np.full(4,p0),np.full(4,p1),np.full(4,.3))
        self.assertAlmostEqual(float(weights@psi),p1-p0,14)
    def test_augmented_input_failures(self):
        for e in [0,1,np.nan]:
            with self.assertRaises(ValueError):augmented_contrast([0,1],[0,1],[.2,.2],[.3,.3],[e,e])
        with self.assertRaises(ValueError):augmented_contrast([0,1],[0,.5],[.2,.2],[.3,.3],[.5,.5])
    def test_clustered_se_independent_loop_oracle(self):
        v=np.array([1.,1.,4.,4.,4.,9.]);g=np.array([3,3,4,4,4,8],np.uint64)
        deviations=[sum(v[g==a]-v.mean()) for a in set(g)]
        expected=np.sqrt(3/2*sum(d*d for d in deviations)/36)
        self.assertAlmostEqual(clustered_summary(v,g)["se"],expected,14)
        self.assertEqual(clustered_summary(v-v,g)["se"],0)
    def test_sensitivity_bayes_table_oracle(self):
        for p in [.001,.03,.3,.999]:
            for r in [.5,1,2,5,10]:
                # Build retained positive and negative mass independently.
                keep0=min(1,1/r);keep1=r*keep0
                positive=p*keep1;negative=(1-p)*keep0
                released=positive/(positive+negative)
                self.assertAlmostEqual(float(original_response(released,r)),p,13)
    def test_sensitivity_boundary_and_monotonicity(self):
        np.testing.assert_array_equal(original_response([0,1],10),[0,1])
        np.testing.assert_allclose(original_response([.2,.7],1),[.2,.7])
        self.assertLess(original_response(.2,10),original_response(.2,2))
        for bad in [0,-1,np.nan,np.inf]:
            with self.assertRaises(ValueError):original_response(.2,bad)
        for bad in [-.1,1.1,np.nan]:
            with self.assertRaises(ValueError):original_response(bad,1)
    def test_covariate_balance_cannot_identify_original_effect(self):
        # Constant X is exactly independent of T, before AND after selection.
        p0,p1=.04,.06
        released0=2*p0/(1-p0+2*p0)
        released1=5*p1/(1-p1+5*p1)
        self.assertGreater(released1-released0,5*(p1-p0))
        self.assertAlmostEqual(float(original_response(released1,5)-original_response(released0,2)),.02,14)
    def test_group_consistent_budget_and_fractional_boundary(self):
        fp=np.array([10,10,20,30],np.uint64);s=np.array([3,3,2,1.])
        a=budget_weights(s,fp,.2)
        np.testing.assert_allclose(a,[.4,.4,0,0])
        self.assertAlmostEqual(a.mean(),.2)
    def test_allocation_row_order_and_negative_net_scores(self):
        fp=np.array([10,10,20,30],np.uint64);s=np.array([3,3,-1,-2.])
        perm=np.array([3,0,2,1])
        np.testing.assert_allclose(budget_weights(s[perm],fp[perm],.7,True),
                                   budget_weights(s,fp,.7,True)[perm])
        self.assertEqual(budget_weights(s,fp,1,True).mean(),.5)
        self.assertEqual(budget_weights(-np.ones(4),fp,.2,True).sum(),0)
    def test_budget_failures(self):
        for b in [-1,1.1,np.nan]:
            with self.assertRaises(ValueError):budget_weights([1,2],[1,2],b)
        with self.assertRaises(ValueError):budget_weights([1,2],[1,1],.2)
    def test_integrated_curve_independent_trapezoid(self):
        fp=np.array([1,2,3,4],np.uint64);s=np.array([4,3,2,1.]);psi=np.array([3,1,0,-2.])
        b=[0,.5,1]
        self.assertAlmostEqual(integrated_contribution(s,fp,psi,b).mean(),.375)
    def test_selection_has_no_test_outcomes_and_fixed_tie_rule(self):
        y=np.array([0,1,0,1]);t=np.array([0,1,1,0]);fp=np.arange(4,dtype=np.uint64)
        pred={"m0":np.full(4,.2),"m1":np.full(4,.3),
              "scores":{k:np.ones(4) for k in ["pooled_logistic","t_learner","dr_learner"]}}
        chosen,_=select_ranking(y,t,fp,pred,np.full(4,.5),[0,.5,1])
        self.assertEqual(chosen,"pooled_logistic")
    def test_fixed_model_training_is_deterministic(self):
        rng=np.random.default_rng(9);x=rng.normal(size=(2400,12))
        y=rng.binomial(1,1/(1+np.exp(x[:,0]+2)))
        a=outcome_model().fit(x,y).predict_proba(x[:100])[:,1]
        b=outcome_model().fit(x,y).predict_proba(x[:100])[:,1]
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-14)

if __name__=="__main__":unittest.main()
