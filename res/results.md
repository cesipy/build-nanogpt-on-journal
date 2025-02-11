# Results

## Run 1
Results with the above parameters: 
```
step   843 | loss: 0.315987 | lr 1.7098e-03 | norm: 0.4969 | dt: 471.28ms | tok/sec: 34764.61
step   844 | loss: 0.267855 | lr 1.7096e-03 | norm: 0.4221 | dt: 471.95ms | tok/sec: 34715.43
step   845 | loss: 0.299901 | lr 1.7093e-03 | norm: 0.5035 | dt: 471.58ms | tok/sec: 34742.46
step   846 | loss: 0.297089 | lr 1.7091e-03 | norm: 0.4651 | dt: 471.80ms | tok/sec: 34726.91
step   847 | loss: 0.313681 | lr 1.7089e-03 | norm: 0.4982 | dt: 471.62ms | tok/sec: 34740.20
step   848 | loss: 0.331955 | lr 1.7086e-03 | norm: 0.4921 | dt: 471.30ms | tok/sec: 34763.20
step   849 | loss: 0.305687 | lr 1.7084e-03 | norm: 0.4941 | dt: 472.50ms | tok/sec: 34675.23
rank 0 sample 0: ## August
seitdem verabredete erst um 10 uhr nach der arbeit musste ich bei barracuda
rank 0 sample 1: ## August ich diese bestellung auszudem gibt sie aber schluss. am nachmittag werde
rank 0 sample 2: ## August ich nicht wirklich down, da
die prüfung war - erlächlich war und meldete mir von
rank 0 sample 3: ## August

heute hatte ich in meinen termin bei gregor - er hat nur in den süß
```
The loss had a steady decrease from about 10 to 0.2 at step 1050. More training resulted in a lower loss. 
The expected loss for a vocab size of about 50k is $-ln(1/50k) = 10.82$. 

Complete training, no fine-tuning!

later with about 3k steps completed: 

```
step  3192 | loss: 0.046291 | lr 6.6593e-04 | norm: 0.1019 | dt: 471.43ms | tok/sec: 34753.62
step  3193 | loss: 0.046444 | lr 6.6545e-04 | norm: 0.1037 | dt: 471.58ms | tok/sec: 34742.58
step  3194 | loss: 0.051136 | lr 6.6498e-04 | norm: 0.1002 | dt: 471.37ms | tok/sec: 34758.47
step  3195 | loss: 0.048103 | lr 6.6450e-04 | norm: 0.1008 | dt: 472.86ms | tok/sec: 34648.73
step  3196 | loss: 0.046616 | lr 6.6403e-04 | norm: 0.0916 | dt: 472.59ms | tok/sec: 34668.50
step  3197 | loss: 0.048399 | lr 6.6355e-04 | norm: 0.1024 | dt: 471.72ms | tok/sec: 34732.29
step  3198 | loss: 0.051926 | lr 6.6308e-04 | norm: 0.1095 | dt: 471.62ms | tok/sec: 34739.67
step  3199 | loss: 0.046581 | lr 6.6260e-04 | norm: 0.1002 | dt: 471.58ms | tok/sec: 34743.08
rank 0 sample 0: ## August
heute hatte ich
ich erstes ps dieses semester; angewandte mathematik; es war
rank 0 sample 1: ## August ich geplant hatte. ursprünglich hatte ich vor, jedoch telefonierte ich
rank 0 sample 2: ## August ich in den letzten tagen habe ich wenig zeit um meine fotos desime zu leben.
rank 0 sample 3: ## August
heute war ich mit lea geburtstag. es war ziemlich fein, jedoch hatte
````
