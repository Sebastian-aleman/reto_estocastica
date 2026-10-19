Sets s especies /LE,SA,SC,ST,OC,OE,OR,OS,PL,YF/
     i nodos /n1*n100/
     E(i,i) aristas;
Alias (s,t), (i,j);
Parameters w(s,t), c(s), p(s), X(s), T, B;
Binary Variable x(i,s);  Positive Variable y(i,j,s,t);
Integer Variable n(s);   Variable z;
Equations fo, unaEsp(i), cant(s), bandaInf(s), bandaSup(s), total, presup, lin(i,j,s,t);
fo..             z =e= sum((i,j,s,t)$E(i,j), w(s,t)*y(i,j,s,t));
unaEsp(i)..      sum(s, x(i,s)) =l= 1;
cant(s)..        sum(i, x(i,s)) =e= n(s);
bandaInf(s)..    X(s) + n(s) =g= 0.9*p(s)*T;
bandaSup(s)..    X(s) + n(s) =l= 1.1*p(s)*T;
total..          sum(s, n(s)) =e= T - sum(s, X(s));
presup..         sum(s, c(s)*n(s)) =l= B;
lin(i,j,s,t)$E(i,j).. y(i,j,s,t) =g= x(i,s) + x(j,t) - 1;
Model reforesta /all/; Solve reforesta using MIP minimizing z;