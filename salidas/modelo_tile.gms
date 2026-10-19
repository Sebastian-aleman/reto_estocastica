* Modelo exacto del patrón toroidal 10x10, formulación fuerte (igual al MILP de PuLP)
* Correr desde la carpeta salidas/ (usa nodos_tile.inc, aristas_tile.inc y W.inc)
Sets s 'especies y vacio' /LE, SA, SC, ST, OC, OE, OR, OS, PL, YF, vacio/
     i 'nodos' /
$include nodos_tile.inc
/
     E(i,i) 'aristas' /
$include aristas_tile.inc
/;
Alias (s,t), (i,j);

* W.inc no trae la fila ni la columna de vacio: GAMS las deja en 0
Table w(s,t)
$include W.inc
;

Parameter n(s) 'nodos por estado (total 100)' /LE 4, SA 20, SC 4, ST 4, OC 5, OE 4, OR 7, OS 6, PL 9, YF 3, vacio 34/;

Binary Variable x(i,s);
Positive Variable y(i,j,s,t);
Variable z;

* Romper simetría por traslación del toroide
x.fx('n1','SA') = 1;

Equations fo, un_estado(i), cantidad(s), enlace_i(i,j,s), enlace_j(i,j,t);
fo..                    z =e= sum((i,j,s,t)$E(i,j), w(s,t)*y(i,j,s,t));
un_estado(i)..          sum(s, x(i,s)) =e= 1;
cantidad(s)..           sum(i, x(i,s)) =e= n(s);
enlace_i(i,j,s)$E(i,j).. sum(t, y(i,j,s,t)) =e= x(i,s);
enlace_j(i,j,t)$E(i,j).. sum(s, y(i,j,s,t)) =e= x(j,t);

Model tile /all/;
option reslim = 900, optcr = 0;
Solve tile using MIP minimizing z;
display z.l, x.l;
