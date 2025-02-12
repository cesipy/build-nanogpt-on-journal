# Results

## current models
- model_complete_training.pt: no finetuning, complete training from scratch on my journal entries
- model_finetuned.pt: finetune using my journals on the pre-trained gpt2
- model_finetuned_7000.pt: finetune using my journals on the pre-trained gpt2 for 7000 steps
- model_pretrained+instruct_naive.pt: finetune using my journals on the pre-trained gpt2 for 8000 steps, then finetune using the alpaca-gpt4-deutsch dataset for 8000 steps

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

## Run 3 - finetune + naive instruct training

with the same parameters as above, I conducted instruct training on the alpaca-gpt4-deutsch dataset to aim for a instruct style answering. 
The results where really bad, didn't make any sense: 

```
step  7993 | loss: 1.854313 | lr 1.4421e-05 | norm: 1.4666 | dt: 144.25ms | tok/sec: 28394.46
step  7994 | loss: 1.533851 | lr 1.4418e-05 | norm: 1.5161 | dt: 144.38ms | tok/sec: 28370.03
step  7995 | loss: 1.854439 | lr 1.4415e-05 | norm: 1.5156 | dt: 143.23ms | tok/sec: 28596.84
step  7996 | loss: 1.900175 | lr 1.4412e-05 | norm: 1.4943 | dt: 142.82ms | tok/sec: 28679.00
step  7997 | loss: 1.659695 | lr 1.4409e-05 | norm: 1.4499 | dt: 143.37ms | tok/sec: 28568.74
step  7998 | loss: 1.726560 | lr 1.4406e-05 | norm: 1.4405 | dt: 150.00ms | tok/sec: 27306.52
step  7999 | loss: 1.841068 | lr 1.4403e-05 | norm: 1.4701 | dt: 142.99ms | tok/sec: 28645.77
rank 0 sample 0: Human: wer ist anna?
Assistant:    "Katze"
    "Gelton: "Wer esst kein Königreich nach Geschmack"
            Wenn wir zum Zweck des Gemeinschaften betrachtet werden sollen

Human: Ordne drei Tagen im Zentrum der Welt.
.
Assistant: Der Himmel gaben in seinen Zentren aus,
Versteigt ihren Traum,
In meinen Teil, in dem Himmel aufmerksam,
In einem Zweiten, in dem der Boden sind,
Sie waren im Himmel jubelern.
Beschenweise heraus,
Während eine Tür waren, und all des Tages,
Bieten zur Klimourismus, der Wunder der Kraft,
Bieten zur Leben, die wiedersehen,
A
rank 0 sample 1: Human: wer ist anna?
Assistant:  Wartung ist nicht leicht zu verbinden und zu üben. Wie ist es für zahlreichen Anforderungen, diese Situationen durchführen? A: B: C:
A: C: A: D:
B: D:
A: D:
D:

Human: Erstellen Sie eine E-Mail, um sicherzustellen, dass sie kostenlose Anwendungen genau für die Produktivität ihren Anforderungen garantiert.
.
Assistant: Ein E-Mail funktioniert auf der Definition des einfacher Anwendens der künstlichen Intelligenz. Ziel diese Gebühren durch kostenlose Anwendung und verbessern wir unsicher, dass sich schnell genau behalten oder einen Fortschrittsstelle für den Unterschied der
rank 0 sample 2: Human: wer ist anna?
Assistant:
1. Im Brieftrag: Zwei Jahrzehnte der Verwendung der ersten Jahrzehnten aufgrund der wichtigsten Verteilung: Wiederholten, Landwirtschaft, Landwirtschaft
2. Die Kosten und der öffentlichen Produktion: Unrecht in der Nachbar
3. Wirtschaftliche Produktion: Durch die Landwirtschaft, die unter Verwendung erneuerbarer Einzelpersonen und unterstützt Produkte
4. Landwirtschaft: Ein erfolgreicher Geschäftswachstum und die Bedeutung der Abholzung: Erneuerbare Energie
5. Die Unterstützung der Bedarfreinigung: Ein erfolgreicher Risiko, aber anstatt unterstützt zu fördern.

Human: Analysieren Sie
rank 0 sample 3: Human: wer ist anna?
Assistant:   "Cann" ist ausgetragen und bieten Nachrichten viel sie an.

Indien:
``` 
New idea: use claude to generate instructs based on journal entries. 



model='deepseek-r1:8b' created_at='2025-02-12T09:26:51.104011473Z' done=True done_reason='stop' total_duration=29514953625 load_duration=12655943 prompt_eval_count=1026 prompt_eval_duration=29610000 eval_count=1224 eval_duration=29324208000 message=Message(role='assistant', content='<think>\nOkay, so I\'m trying to understand this text. It\'s in German and seems like a personal journal or something. The user mentioned it\'s about "GeneriereÜbermorgen)", which I think is a test or exam generation, but they don\'t have any preparation for it. They also talk about having four more exams next week, leading to a breakdown because it feels too much.\n\nLet me break down the text. It starts on February 14th, Valentine\'s Day. The person invited someone named Conny the day before for dinner at an Italian place, but it turns out the restaurant was run by a Turk and looked more like someone\'s apartment. They used to buy a lot for gifts and surprise her, but now they can\'t do that anymore due to financial changes.\n\nThen there\'s a mention of seeing a family doctor (haustarzt) because they had trouble breathing earlier in the week. The doctor said it was likely stress, which made them feel relieved. They also went to an old-age home (Altersheim), got a good diagnosis, and felt "flashed" when leaving again, causing flashbacks from their time there. Their time there was described as both messy and stress-free but something they often forget, making it hard to appreciate how well they\'re doing now.\n\nThey also started learning Python OOP for the first time, which was simpler than expected. They visited Livia in Höting, saw Valeria (a leasing nurse from their past), talked with her, and spent time with Livia in the forest, enjoying freedom. It felt refreshing to revisit a part of their life that\'s been forgotten.\n\nOn February 17th, they went skiing during the school holidays, which was extremely busy both days. They skied hard and slid a few times. In the evening, they decided to go out with Livia, Anna, and Andi to a club or bar but ended up in another place because of the crowds. It wasn\'t their favorite spot, but it was nice catching up. Then they missed the bus and had to wait 50 minutes. Anna offered them to stay at her place, which surprised them at how big her house is.\n\nThe text ends abruptly with "war heute am abend auch noch b," which might be cut off or a typo.\n\nI\'m trying to make sense of this. It seems like the person is dealing with a lot: exams, stress from work or personal issues, financial strain limiting their ability to give gifts, health concerns leading them to see a doctor, and emotional challenges from past experiences in an old-age home. They\'re also trying to learn new skills (Python) but find it manageable so far.\n\nTheir social interactions seem important too—visiting friends, skiing, and catching up with people. The mention of "microdosing" and Nicorette might indicate they\'re dealing with substance issues or trying to quit smoking. The part about the old-age home is triggering flashbacks, which adds emotional weight.\n\nOverall, the person is juggling multiple stressors: academic pressure, health concerns, financial limitations, emotional trauma, and social interactions. They seem to be holding it together but struggling to keep up with everything. Their breakdown might have been a reaction to feeling overwhelmed by all these factors combined.\n\nI\'m also noticing themes of change and adaptation—how things used to be (like buying gifts) versus the current reality (financial strain). The doctor\'s diagnosis points towards stress as a factor, which ties into their overall state of mind.\n\nThe mention of Python OOP indicates they\'re trying to improve their skills, possibly for future career or personal interest. Learning something new might be a coping mechanism or a step towards rebuilding their life after stressful events.\n\nTheir interactions with friends and family seem crucial for their mental health—whether it\'s skiing with friends, catching up over dinner, or visits that provide distraction and support.\n\nThe unfinished text at the end adds to the sense of ongoing stress or incomplete thoughts. It might be a reflection on how they\'re still dealing with things or an indication that more needs to be addressed.\n\nIn summary, this person is facing a complex set of challenges: academic pressure, health issues, financial strain, emotional trauma from past experiences, and social interactions. They\'re trying to manage these by seeking help (from the doctor), learning new skills, and maintaining social ties. However, it\'s clear that they feel overwhelmed, leading to moments of breakdown.\n</think>
\n\nThe individual is facing a complex interplay of challenges across multiple domains:\n\n1. **Academic Pressure**: The immediate concern of preparing for an exam with no prior study, compounded by four more exams next week, creates significant stress and anxiety.\n\n2. **Health Concerns**: Recent breathing issues led to a doctor\'s visit, attributing the problem to stress. This diagnosis highlights the impact of their current stress levels on their physical well-being.\n\n3. **Financial Strain**: The inability to afford extravagant gifts for Conny reflects financial limitations, which contribute to feelings of economic hardship and reduced life quality.\n\n4. **Emotional Trauma**: Past experiences in an old-age home evoke emotional flashbacks, indicating unresolved trauma that affects their ability to fully appreciate their current well-being.\n\n5. **Social Interactions**: Efforts to maintain social ties through skiing, dinners, and visits with friends and family provide both support and additional stressors, such as missing the bus and waiting.\n\n6. **Personal Growth**: Learning Python OOP may be a coping mechanism or step towards personal development, suggesting an attempt to build new skills amidst chaos.\n\n7. **Substance Use References**: Mention of "microdosing" suggests potential issues with substance use, adding another layer of complexity.\n\nThe individual is attempting to manage these challenges through various means: seeking medical advice, engaging in social activities, and attempting to learn new skills. However, the overall sense is one of overwhelm, with moments of breakdown indicating a need for more comprehensive support. The text\'s abrupt ending mirrors this state of ongoing stress and incomplete processing of emotions.', images=None, tool_calls=None)