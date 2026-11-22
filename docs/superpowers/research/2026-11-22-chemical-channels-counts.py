def lcount(k, lmax, D):
    return 0 if k > D else min(lmax, D-k)+1
def p3(K, nmax=8, lmax=3, D=4):
    o = sum(lcount(a+b,lmax,D) for a in range(nmax) for b in range(nmax))
    d = sum(lcount(2*n,lmax,D) for n in range(nmax))
    return (K*K*o + K*d)//2
def p4(K, nmax=8, lmax=3, D=4, d4=3):
    d = min(D, d4)
    ltop = min(lmax, 2*d)
    items = [(t,n,l) for t in range(K) for n in range(min(nmax,d+1)) for l in range(ltop+1)]
    m = len(items); c = 0
    for a in range(m):
        t1,n1,l1 = items[a]
        for b in range(a,m):
            t2,n2,l2 = items[b]
            if n1+n2+(l1+l2)//2 > d: continue
            for (t3,n3,l3) in items[b:]:
                s = l1+l2+l3
                if s%2==1 or l3>l1+l2 or l3+min(l1,l2) < max(l1,l2): continue
                if n1+n2+n3+s//2 > d: continue
                c += 1
    return c
def qsp(S, **kw):
    d4 = kw.pop('d4',3)
    return 8*S*(S+1)//2 + S*(p3(S,**kw)+p4(S,d4=d4,**kw))
def qch(S, K, **kw):
    d4 = kw.pop('d4',3)
    K = min(K,S)
    return 8*S*(S+1)//2 + S*(p3(K,**kw)+p4(K,d4=d4,**kw))
for D in (4,5,6):
  for d4 in (2,3):
    print("D",D,"d4",d4,"species S=1..6:", [qsp(S,D=D,d4=d4) for S in range(1,7)])
print()
for D,d4 in ((4,3),(6,3)):
  print("D",D,"d4",d4)
  for K in (2,3,4,5,6):
    print("  K",K,"P3",p3(K,D=D),"P4",p4(K,D=D,d4=d4),"Q' S=4..12:", [qch(S,K,D=D,d4=d4) for S in (4,5,6,8,10,12)])
# low-order fit for learning W: body2 + body3 species
print()
for (nm,lm,D) in ((8,3,4),(4,2,2),(6,2,3)):
  print("low-order nmax",nm,"lmax",lm,"D",D,[ (S, 8*S*(S+1)//2 + S*p3(S,nmax=nm,lmax=lm,D=D)) for S in (5,8,10,12,15)])
