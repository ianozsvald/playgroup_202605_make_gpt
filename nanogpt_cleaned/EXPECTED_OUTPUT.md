# Expected output from a trained model

## CPU (smaller model), manual attn (default in our code)

```
$ python sample.py 
WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0
WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0
WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0
WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0
number of parameters: 0.80M
Loading meta from shakespeare_char/meta.pkl...

I by done what leave death,
And aproposely beef the are and sors blate though wat our fort
Thine the aftior than whating bods farse dowed
And nears and thou stand murs's consel.

MEOF:
Sir, should and then thee.
```

## CPU (smaller model), Flash attention (disabled in our code)

```
$ python sample.py --out_dir=shakespeare_char
Overriding: out_dir = shakespeare_char
number of parameters: 0.80M
Loading meta from shakespeare_char/meta.pkl...

When becond whow and is by bet
baide a way dongant One my dained
he arthip foul him the are that ane away, my thand
a wizok me and the of in he coursiend;
Whoues is enden minelation in overs,
He will now ould and well in you liel?

CORIOLANN:
Sup; and thew you love us you pater to whom
Ist the to killo Wind to do evicks to them,
Will we high he poopte the the deked nonrury for trear.
Why:
Ey, I may post that Prove my of that but
Hartiond a wards be his a forth in course.

ANGHERW:
My have for hi
```

## GPU (bigger, slower model)

```
$ python sample.py --out_dir=shakespeare_char 
Overriding: out_dir = shakespeare_char
number of parameters: 10.65M
Loading meta from shakespeare_char/meta.pkl...


Clown:
So you will be a fellow a serpician well.

OXFORD:
Sir, I saw this foul command.

Servant:
I am court.

PRINCE EDWARD:
You shall not hear that now I was mine.

CLIFFORD:
Your liege, sir.
```
