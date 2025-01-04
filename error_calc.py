import re as regex
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from sympy import *
from sympy.core import Expr as sympy_Expr

unicode_to_latex = {
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "ο": r"o",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    "φ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "Α": r"A",
    "Β": r"B",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Ε": r"E",
    "Ζ": r"Z",
    "Η": r"H",
    "Θ": r"\Theta",
    "Ι": r"I",
    "Κ": r"K",
    "Λ": r"\Lambda",
    "Μ": r"M",
    "Ν": r"N",
    "Ξ": r"\Xi",
    "Ο": r"O",
    "Π": r"\Pi",
    "Ρ": r"P",
    "Σ": r"\Sigma",
    "Τ": r"T",
    "Υ": r"\Upsilon",
    "Φ": r"\Phi",
    "Χ": r"X",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
}

unicode_to_latex_pattern = regex.compile("|".join(regex.escape(key) for key in unicode_to_latex.keys()))


class PartialDerivative(Derivative):
    pass


def replace_greek_with_latex(input_string):
    return unicode_to_latex_pattern.sub(lambda match: f"{unicode_to_latex[match.group(0)]} ", input_string)


@contextmanager
def latex_context(filename="equation.tex"):
    if filename is None:
        yield lambda x: None, lambda x: None
        return
    try:
        with open(filename, "w") as f:
            f.write(r"\documentclass{article}")
            f.write("\n")
            f.write(r"\usepackage{amsmath}")
            f.write("\n")
            f.write(r"\usepackage{breqn}")
            f.write("\n")
            f.write(r"\begin{document}")
            f.write("\n")

            def writer(expr):
                f.write(f"\\[{replace_greek_with_latex(latex(expr))}\\]")
                f.write("\n")

            def writer_breqn(expr):
                f.write(r"\begin{dmath}")
                f.write("\n")
                f.write(f"{replace_greek_with_latex(latex(expr))}")
                f.write("\n")
                f.write(r"\end{dmath}")
                f.write("\n")

            yield writer, writer_breqn
            f.write(r"\end{document}")
    except Exception as e:
        traceback.print_exc()


def _flatten_expr(expr: sympy_Expr, partials: dict[Symbol, sympy_Expr], measurables: dict[Symbol, float]):
    for fs in sorted(list(expr.free_symbols), key=lambda s: str(s)):
        if fs not in measurables:
            for q in _flatten_expr(partials[fs], partials, measurables):
                expr = expr.subs(fs, q)
    yield expr


def flatten_expr(expr: sympy_Expr, partials: dict[Symbol, sympy_Expr], measurables: dict[Symbol, float]):
    return next(_flatten_expr(expr, partials, measurables))


def do_error_calc(target_var: Symbol, target_expr: sympy_Expr, partials: dict[Symbol, sympy_Expr], measurables: dict[Symbol, float], outtex: Path):
    """Symbolic error calculation by propagating derivatives

    Args:
        target_var (Symbol): The thing you are trying to compute with our measurements
        target_expr (sympy_Expr): The expression that describes your thing.
        partials (dict[Symbol, sympy_Expr]): A dictionary that maps symbols in target_expr to a corresponding
            expression. For example let target_expr = d**2+a, and 'a' is measureable but to determine 'd' you
            calculate sqrt(x**2+y**2), with x and y being measurable, then you would set
            partials={d:sqrt(x**2+y**2)}. Notice that since 'a' is measurable, there are no partials for 'a'.
        measurables (dict[Symbol, float]): Measureable variables and the accociated error.
        outtex (Path): Filename for .tex output. This overwrites the file!

    Example:
    g, h, x, α_rad, t, x_coord, x_start, z_coord, z_start, α_deg = symbols("g h x \\alpha_{rad} t x_{coord} x_{start} z_{coord} z_{start} \\alpha_{deg}")
    dodo(
        target_var=g,
        target_expr=2 * (h + x * tan(α_rad)) / t**2,
        partials={x: sqrt((x_coord - x_start) ** 2 + (z_coord - z_start) ** 2), α_rad: α_deg * pi / 180},
        measurables={x_coord: 0.0005, x_start: 0.0005, z_coord: 0.0005, z_start: 0.0005, α_deg: 0.0005, t: 0.5, h: 0},
        outtex=Path("equation.tex")
    )

    """
    with latex_context(filename=outtex) as (display, display_breqn):
        print(f"Measure vars:")
        print(f"    {measurables}")
        print(f"Target expression:")
        print(f"    {target_var} = {target_expr}")

        missing_partial_expressions = [free_symbol for free_symbol in target_expr.free_symbols if free_symbol not in partials and free_symbol not in measurables]
        if len(missing_partial_expressions) > 0:
            raise Exception(f"Missing partial expressions for {pretty(missing_partial_expressions)}")

        def foo(variable, expr: sympy_Expr):
            variable_err = symbols(f"Δ{variable}")
            l = []
            t = []
            some_d = {}
            to_display = []

            display(Eq(variable, expr, evaluate=False))
            for fs in sorted(list(expr.free_symbols), key=lambda s: str(s)):
                fs_err = symbols(f"Δ{fs}")
                derivative = diff(expr, fs)
                l.append(abs(derivative) * fs_err)
                t.append(abs(PartialDerivative(variable, fs)) * fs_err)
                # display(Eq(fs_err * abs(PartialDerivative(variable, fs)), fs_err * abs(derivative), evaluate=False))
                if fs in measurables:
                    some_d[fs_err] = measurables[fs]
                    to_display.append(Eq(fs_err, measurables[fs], evaluate=False))
                    # display(Eq(fs_err, measure_vars[fs], evaluate=False))
                else:
                    yield from foo(fs, partials[fs])
            # display(Eq(variable_err, sum(t), evaluate=False))
            for _to_display in to_display:
                display(_to_display)
            display(Eq(variable_err, sum(l), evaluate=False))
            yield [Eq(variable_err, sum(t), evaluate=False), Eq(variable_err, sum(l), evaluate=False)]

        r = list(foo(target_var, target_expr))
        full = r.pop(-1)
        for q in r[::-1]:
            full[0] = full[0].subs(q[0].lhs, q[0].rhs)
            full[1] = full[1].subs(q[1].lhs, q[1].rhs)
        display_breqn(Eq(full[0].lhs, full[0].rhs, evaluate=True))
        display_breqn(Eq(full[1].lhs, full[1].rhs, evaluate=True))
        some_d = {}
        for k, v in measurables.items():
            some_d[symbols(f"Δ{k}")] = v
        error_eq = Eq(full[1].lhs, full[1].rhs.subs(some_d), evaluate=False)
        error_eq_formal = full[0]
        return error_eq_formal, error_eq


if __name__ == "__main__":
    g, h, x, α_rad, t, x_coord, x_start, z_coord, z_start, α_deg = symbols("g h x \\alpha_{rad} t x_{coord} x_{start} z_{coord} z_{start} \\alpha_{deg}")
    g_expr = 2 * (h + x * tan(α_rad)) / t**2
    g_partials = {x: sqrt((x_coord - x_start) ** 2 + (z_coord - z_start) ** 2), α_rad: α_deg * pi / 180}
    g_measurables = {x_coord: 0.0005, x_start: 0.0005, z_coord: 0.0005, z_start: 0.0005, α_deg: 0.0005, t: 0.5, h: 0}
    _, g_err_eq = do_error_calc(
        target_var=g,
        target_expr=g_expr,
        partials=g_partials,
        measurables=g_measurables,
        outtex=Path("equation_g.tex"),
        # outtex=None,
    )

    v = symbols("v")
    v_expr = x * sqrt(g / (h * cos(2 * α_rad) + h + x * sin(2 * α_rad)))
    v_partials = {x: sqrt((x_coord - x_start) ** 2 + (z_coord - z_start) ** 2), α_rad: α_deg * pi / 180}
    v_measurables = {x_coord: 0.0005, x_start: 0.0005, z_coord: 0.0005, z_start: 0.0005, α_deg: 0.0005, h: 0, g: 0.0008}
    _, v_err_eq = do_error_calc(
        target_var=v,
        target_expr=v_expr,
        partials=v_partials,
        measurables=v_measurables,
        # outtex=Path("equation_v.tex"),
        outtex=None,
    )

    def parse_excel(measurements):
        return [[float(y) for y in x.strip().split("\t")] for x in measurements.strip().split("\n") if x.strip()]

    def evaluate_measurements(expr, err_eq, partials, measurables, measurements, decimals=4):
        flattened_expr = flatten_expr(expr, partials, measurables)
        flattened_err_expr = flatten_expr(err_eq.rhs, partials, measurables)
        vals = []
        errs = []
        print(f"{"Evaluating measurements"}          (computation time)")
        i = 1
        n_measurements = len(measurements)
        __elapsed_list = []
        for m in measurements:
            α_deg_val = m[0]
            x_coord_val = m[1]
            z_coord_val = m[2]
            x_start_val = 0.5
            z_start_val = 0.5
            h_val = 1.6
            g_val = 0.0239
            t_val = m[3]
            params = {α_deg: α_deg_val, x_coord: x_coord_val, z_coord: z_coord_val, x_start: x_start_val, z_start: z_start_val, h: h_val, g: g_val, t: t_val}
            __start_time = time.time()
            val = round(flattened_expr.subs(params).evalf(), decimals)

            err_val = round(flattened_err_expr.subs(params).evalf(), decimals)
            __elapsed = time.time() - __start_time
            vals.append(val)
            errs.append(err_val)
            s = f"{i}/{n_measurements}"
            __elapsed_list.append(__elapsed)
            print(f"#{s:8}    {val:.{decimals}f}±{err_val:.{decimals}f}       ({round(__elapsed, 4)}s)")
            i += 1
        val_mean = round(sum(vals) / len(vals), decimals)
        err_mean = round(sum(errs) / len(errs), decimals)
        print(f"mean: {val_mean:.{decimals}f}±{err_mean:.{decimals}f} {round(100*err_mean/val_mean, 2)}%              ({round(sum(__elapsed_list), 4):.4f}s)")

    evaluate_measurements(
        g_expr,
        g_err_eq,
        g_partials,
        g_measurables,
        parse_excel(
            """
            85	0.722	7.549	91
            75	0.28	22.437	87
            65	0.444	35.404	83
            55	1.708	46.407	76
            45	0.773	51.912	67
            35	0.657	52.607	56
            25	-0.174	48.113	44
            15	-0.067	37.583	30
            5	0.333	21.463	16
            0	0.437	14.811	11"""
        ),
    )
