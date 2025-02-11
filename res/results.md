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


## Run 2- finetune

in this run I fine-tuned the gpt2 weights on my journal entries. I used the following params: 
```python
total_batch_size = 4096 # 2**19, ~0.5M, in number of tokens
print(ddp_world_size)
B = 8 # micro batch size
T = 512 # sequence length

max_lr = 3e-5 * 3
min_lr = max_lr * 0.1
warmup_steps = 100
max_steps = 5000
```

The results were decent, but dont have much in commom with my journal entries. 
It was trained for about 3500 steps

```
step  3543 | loss: 0.071956 | lr 2.5422e-05 | norm: 0.8227 | dt: 144.11ms | tok/sec: 28423.26
step  3544 | loss: 0.076550 | lr 2.5402e-05 | norm: 0.8325 | dt: 144.09ms | tok/sec: 28425.80
step  3545 | loss: 0.115621 | lr 2.5381e-05 | norm: 1.1536 | dt: 144.04ms | tok/sec: 28436.67
step  3546 | loss: 0.088372 | lr 2.5360e-05 | norm: 0.9180 | dt: 144.18ms | tok/sec: 28409.30
step  3547 | loss: 0.106261 | lr 2.5339e-05 | norm: 1.0779 | dt: 143.74ms | tok/sec: 28495.01
step  3548 | loss: 0.083351 | lr 2.5318e-05 | norm: 0.8445 | dt: 144.56ms | tok/sec: 28333.49
step  3549 | loss: 0.105626 | lr 2.5297e-05 | norm: 1.1120 | dt: 144.45ms | tok/sec: 28356.78
rank 0 sample 0: ## August
für mich war franz dann auch als wir knapp 40km/h. es war leider auch BALSE, aber nicht auch das komplett ausschaut. fühlte mich schon wieder gehen wollen! dann schaue ich den bus nur 30-ingers tour. bin ich überzeugt von lea, dass ich das gefühl von gell in mein gedanken bin und mein wichtiges an diese zeit brauchen würde.


## August 25
heute machte ich mit franz essen, bei franz gehen. dabei waren teilweise viel zu bewusst, aber teilweise hören wir mit ihr franz bei franz kommen. war sehr passion und sagte ein langsam uns auf der sowi bib leisten.

in der zweiten hä
rank 0 sample 1: ## August
die letzte woche sehr interessantrach war, hatte rust wieder buch gegen 12 uhr.
Gemeinsam radelten wir zur 11 au in der KI. wir wurden beim 19.5 angesprochen. wir redeten viel und die ganze zeit in das projekt.

heute abend las ich das damit, ich bekam zu entspannen!


## April 11

habe heute den ganzen tag auf zu gerat. ich konnte in den stunden nicht gut über dinge. dafür stunden ziemlich über meine neue traum!). höllische transformation scheiss ich in jade. das war recht gut. so wird es mir ein großes bild über körper, die ricthaige zugfahr war.
rank 0 sample 2: ## August ich muss zugeben, dass unsere beziehung passt und von ihren gebieten alsen wäre:) leider gebe ich oft das nochBasicVariables können, ob ich michinchenbei mache:)] musste ich auch innsbruck nicht den kopf vergleichen.

<img src="./res/flug1.jpg" width=500>

Mir sägo ist ziemlich chillig, ich wollte ihr mann gefragt, als mir eine republik von ihr leben. inzwisculi war ich oft definitiv mehr. wir hat inzwischen andere typen. zum einen, das uns magdalena und anderen weiter bei wegständen ist sehr emotional enttäuscht. Die wunderschöneine reichten wir mit ihr am weg. Mit Froni, das haben wirklich eine diverse in der fra
rank 0 sample 3: ## August
die letzte woche war 01.30 losfahren. mama war ich auch keinen sohn.

papa hat mir wirklich "heute arbeitstag". ich muss noch etwas setze und in der letztenpykte war noch durch die arbeitstagnis konzept (??). heute war ich etwas später auf passen vollkommen überwarti stellen.

am morgen waren wir das erste mal nach absam und verstand werden mit fragen zusammen.

## April 7

habe heute mit mama den film magnolia angesehen und muss sagen, es war ein wirklich gute argument. Es war ein modelle verwundert, aber es war mega bekomme.

papa macht es sich nicht traurig, war mega schön.

ab
```