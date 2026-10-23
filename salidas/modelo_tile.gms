* Modelo exacto del patrón toroidal 10x10 sin vacíos, formulación fuerte (igual al MILP de PuLP)
* Correr desde la carpeta salidas/ (usa nodos_tile.inc, aristas_tile.inc y W.inc)
Sets s 'especies' /LE, SA, SC, ST, OC, OE, OR, OS, PL, YF/
     i 'nodos' /
$include nodos_tile.inc
/
     E(i,i) 'aristas' /
$include aristas_tile.inc
/;
Alias (s,t), (i,j);

Table w(s,t)
$include W.inc
;

Parameter n(s) 'plantas por especie (total 100 = todos los nodos)' /LE 6, SA 30, SC 6, ST 6, OC 8, OE 6, OR 11, OS 10, PL 13, YF 4/;

Binary Variable a(i,s);
Positive Variable y(i,j,s,t);
Variable z;

* Romper simetría por traslación del toroide
a.fx('n1','SA') = 1;

Equations fo, una_especie(i), cantidad(s), enlace_i(i,j,s), enlace_j(i,j,t);
fo..                    z =e= sum((i,j,s,t)$E(i,j), w(s,t)*y(i,j,s,t));
una_especie(i)..        sum(s, a(i,s)) =e= 1;
cantidad(s)..           sum(i, a(i,s)) =e= n(s);
enlace_i(i,j,s)$E(i,j).. sum(t, y(i,j,s,t)) =e= a(i,s);
enlace_j(i,j,t)$E(i,j).. sum(s, y(i,j,s,t)) =e= a(j,t);

Model tile /all/;
option reslim = 900, optcr = 0;
Solve tile using MIP minimizing z;
display z.l, a.l;
