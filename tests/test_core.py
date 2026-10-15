import unittest
import numpy as np
from customer_policy.data import split_indices, FEATURES
from customer_policy.evaluation import augmented_scores, policy_scores, allocate, summarize

class CausalEvaluationTests(unittest.TestCase):
    def test_augmented_score_known_case(self):
        y = np.array([10., 20., 30.])
        a = np.array([0, 1, 2])
        mu = np.full((3, 3), 5.)
        expected = np.array([[20.,5.,5.],[5.,50.,5.],[5.,5.,80.]])
        np.testing.assert_allclose(augmented_scores(y,a,mu), expected)

    def test_randomization_recovers_effect_with_wrong_nuisance(self):
        rng = np.random.default_rng(17)
        n = 100000
        a = rng.integers(0,3,n)
        x = rng.normal(size=n)
        y = 4*x + np.choose(a,[0.,2.,-1.]) + rng.normal(size=n)
        g = augmented_scores(y,a,np.full((n,3),99.))
        estimate = summarize(g[:,1]-g[:,0])
        self.assertLess(abs(estimate['estimate']-2.),4*estimate['standard_error'])

    def test_budget_picks_value_and_rejects_negative(self):
        means=np.array([[0,4,2],[0,0,-1],[0,2,5],[0,1,0]],float)
        actions=allocate(means,1,.5,.5,np.arange(4))
        np.testing.assert_array_equal(actions,[1,0,2,0])
        np.testing.assert_array_equal(allocate(means,1,10,1,np.arange(4)),np.zeros(4,int))

    def test_zero_capacity(self):
        np.testing.assert_array_equal(allocate(np.ones((7,3)),.4,0,0,np.arange(7)),np.zeros(7))

    def test_split_independence_and_coverage(self):
        split=split_indices(64000,20261015)
        self.assertEqual([len(split[k]) for k in ['train','development','test']],[38400,12800,12800])
        self.assertEqual(len(np.unique(np.concatenate(list(split.values())))),64000)
        self.assertFalse(set(FEATURES)&{'segment','visit','conversion','spend'})

    def test_identical_policies_have_zero_paired_difference(self):
        rng=np.random.default_rng(4);g=rng.normal(size=(100,3));a=rng.integers(0,3,100)
        diff=policy_scores(g,a,.4,.05)-policy_scores(g,a,.4,.05)
        self.assertEqual(summarize(diff)['standard_error'],0)

    def test_invalid_propensity_rejected(self):
        with self.assertRaises(ValueError): augmented_scores(np.zeros(2),np.zeros(2,int),np.zeros((2,3)),[0,.5,.5])


class RobustnessTests(unittest.TestCase):
    def test_null_effect_is_not_manufactured_by_augmentation(self):
        rng=np.random.default_rng(23);n=60000
        a=rng.integers(0,3,n);x=rng.normal(size=n);y=3*x+rng.normal(size=n)
        wrong=np.column_stack([x+2,x-4,x+7])
        score=augmented_scores(y,a,wrong)
        contrast=summarize(score[:,1]-score[:,0])
        self.assertLess(abs(contrast['estimate']),4*contrast['standard_error'])

    def test_policy_recovers_known_heterogeneous_value(self):
        rng=np.random.default_rng(54);n=120000
        x=rng.choice([-1.,1.],n);a=rng.integers(0,3,n)
        potential=np.column_stack([np.zeros(n),2*x,-x])
        y=potential[np.arange(n),a]+rng.normal(size=n)
        action=np.where(x>0,1,2)
        gamma=augmented_scores(y,a,np.zeros((n,3)))
        result=summarize(policy_scores(gamma,action))
        truth=potential[np.arange(n),action].mean()
        self.assertLess(abs(result['estimate']-truth),4*result['standard_error'])

    def test_feature_transform_ignores_assigned_arm_and_outcomes(self):
        import pandas as pd
        from customer_policy.models import encoder
        d=pd.DataFrame({'recency':[1,2,3],'history':[20,30,40],'mens':[1,0,1],'womens':[0,1,1],'newbie':[0,1,0],'zip_code':['Urban','Rural','Urban'],'channel':['Web','Phone','Web'],'segment':['No E-Mail']*3,'visit':[0,1,0],'conversion':[0,0,0],'spend':[0,0,0]})
        e=encoder();x=e.fit_transform(d)
        d.loc[:,'spend']=999999;d.loc[:,'segment']='Mens E-Mail';d.loc[:,'conversion']=1
        np.testing.assert_array_equal(x,e.transform(d))

if __name__=='__main__':unittest.main()
