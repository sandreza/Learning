using Gnuplot

###
@gp 1:20
###
x = LinRange(-2pi, 2pi, 1000)
@gp t="Polynomial approximation of sin(x)"  "set style fill transparent solid 0.6 noborder"
@gp :- raw"""set xtics ('$-\pi$' -pi, '$-\pi/2$' -pi/2, 0, '$\pi/2$' pi/2, '$\pi$' pi)"""
@gp :- xr=3.8.*[-1, 1] yr=[-1.5,1.5] key="box opaque left horiz" linetypes(:Blues_3) "set grid front"
latex = raw"""\begin{minipage}[c]{\textwidth}\begin{equation*}""" *
	raw"""\sin(x) = \sum_0^{+\infty} \frac{(-1)^n}{(2n + 1)!} x^{2n+1}""" *
	raw"""\end{equation*} \end{minipage}"""
@gp :- "set label at graph 0.62,0.2 front center '$latex'"
approx = fill(0., length(x));
@gp :- x sin.(x) approx .+=  x          "w filledcurve t 'n=0' lt 1"
@gp :- x sin.(x) approx .+= -x.^3/6     "w filledcurve t 'n=1' lt 2"
@gp :- x sin.(x) approx .+=  x.^5/120   "w filledcurve t 'n=2' lt 3"
@gp :- x sin.(x) approx .+= -x.^7/5040  "w filledcurve t 'n=3' lt 4"
@gp :- x sin.(x)                        "w l t 'sin(x)' lw 2 lc rgb 'black'"
save(term="cairolatex pdf input color dashed size 5in,3.3in", output="test.tex")
