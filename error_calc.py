from __future__ import annotations

import re as regex
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from pprint import pformat as pprint_pformat

import numba as nb
import numpy as np
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


def wrap_arg(arg_index: int, wrapper_func):
    def decorator(func):
        def wrapped(*args, **kwargs):
            args = list(args)
            if arg_index < len(args):
                args[arg_index] = wrapper_func(args[arg_index])
            return func(*args, **kwargs)

        return wrapped

    return decorator


def wrap_kwarg(kwarg: str, wrapper_func):
    def decorator(func):
        def wrapped(*args, **kwargs):
            if kwarg not in kwargs:
                raise Exception(f"kwarg {kwarg} not in {kwargs} of func {func}")
            kwargs[kwarg] = wrapper_func(kwargs[kwarg])
            return func(*args, **kwargs)

        return wrapped

    return decorator


class Measureables(dict[tuple[Symbol, str], np.float64]):
    @classmethod
    def cast(cls, any_dict: dict) -> Measureables:
        if type(any_dict) == Measureables:
            return any_dict
        elif all(type(k) == Symbol and type(v) == float or type(v) == int for k, v in any_dict.items()):
            return cls.from_symbol_dict(any_dict)
        else:
            raise NotImplementedError(f"{type(any_dict)} cannot be cast to Measureables")

    @classmethod
    def from_symbol_dict(cls, symbol_dict: dict[Symbol, float]) -> Measureables:
        measurables = cls()
        for measurable_symbol, measurable_error in symbol_dict.items():
            measurable_name = str(measurable_symbol).replace("{", "").replace("}", "")
            measurables[(measurable_symbol, measurable_name)] = np.float64(measurable_error)
        return measurables

    def get_str_for_symbol_key(self, symbol_key: Symbol):
        for k in self.keys():
            if k[0] == symbol_key:
                return k[1]
        return None

    def itersymbols(self):
        for k, v in self.items():
            yield (k[0], v)

    def iterstrings(self):
        for k, v in self.items():
            yield (k[1], v)

    def __contains__(self, key):
        if isinstance(key, Symbol):
            return any(k[0] == key for k in self.keys())
        elif isinstance(key, str):
            return any(k[1] == key for k in self.keys())
        return super().__contains__(key)

    def __getitem__(self, key):
        if isinstance(key, Symbol):
            for k in self.keys():
                if k[0] == key:
                    return super().__getitem__(k)

        elif isinstance(key, str):
            for k in self.keys():
                if k[1] == key:
                    return super().__getitem__(k)
        if key not in self:
            print(f"Key '{key}' not found, returning None")
            return None
        return super().__getitem__(key)

    def __repr__(self):
        return pprint_pformat(dict(self.iterstrings()))


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


def _flatten_expr(expr: sympy_Expr, partials: dict[Symbol, sympy_Expr], measurables: Measureables):
    for fs in sorted(list(expr.free_symbols), key=lambda s: str(s)):
        if fs not in measurables:
            for q in _flatten_expr(partials[fs], partials, measurables):
                expr = expr.subs(fs, q)
    yield expr


def flatten_expr(expr: sympy_Expr, partials: dict[Symbol, sympy_Expr], measurables: Measureables):
    return next(_flatten_expr(expr, partials, measurables))


def foo(variable, expr: sympy_Expr, measurables: Measureables, partials: dict[Symbol, sympy_Expr], display):
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
            yield from foo(fs, partials[fs], measurables, partials, display)
    # display(Eq(variable_err, sum(t), evaluate=False))
    for _to_display in to_display:
        display(_to_display)
    display(Eq(variable_err, sum(l), evaluate=False))
    yield [Eq(variable_err, sum(t), evaluate=False), Eq(variable_err, sum(l), evaluate=False)]


@wrap_kwarg("measurables", Measureables.cast)
def do_error_calc(target_var: Symbol = None, target_expr: sympy_Expr = None, partials: dict[Symbol, sympy_Expr] = None, measurables: dict[Symbol, float] | Measureables = None, outtex: Path = None):
    """Symbolic error calculation by propagating derivatives

    Args:
        target_var (Symbol): The thing you are trying to compute with our measurements
        target_expr (sympy_Expr): The expression that describes your thing.
        partials (dict[Symbol, sympy_Expr]): A dictionary that maps symbols in target_expr to a corresponding
            expression. For example let target_expr = d**2+a, and 'a' is measureable but to determine 'd' you
            calculate sqrt(x**2+y**2), with x and y being measurable, then you would set
            partials={d:sqrt(x**2+y**2)}. Notice that since 'a' is measurable, there are no partials for 'a'.
        measurables (dict[Symbol, float] | Measureables): Measureable variables and the accociated error.
        outtex (Path): Filename for .tex output. This overwrites the file!

    Example:
    g, h, x, α_rad, t, x_coord, x_start, z_coord, z_start, α_deg = symbols("g h x \\alpha_{rad} t x_{coord} x_{start} z_{coord} z_{start} \\alpha_{deg}")
    do_error_calc(
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

        r = list(foo(target_var, target_expr, measurables, partials, display))
        full = r.pop(-1)
        for q in r[::-1]:
            full[0] = full[0].subs(q[0].lhs, q[0].rhs)
            full[1] = full[1].subs(q[1].lhs, q[1].rhs)
        display_breqn(Eq(full[0].lhs, full[0].rhs, evaluate=True))
        display_breqn(Eq(full[1].lhs, full[1].rhs, evaluate=True))
        some_d = {}
        for k, v in measurables.itersymbols():
            some_d[symbols(f"Δ{k}")] = v
        print(some_d)
        return full[0].rhs, full[1].rhs.subs(some_d)


def generate_measureables_tui() -> dict[tuple[Symbol, str], np.float64]:
    print("Enter measureable variables and the associated measure errors as a tuple: Name, Error")
    measurables_strings = {}
    while True:
        inp = input(">")

        if not inp:
            print(f"{len(measurables_strings)} measurables entered.")
            break
        else:
            sp = inp.split(", ")
            if len(sp) != 2:
                sp = inp.split(",")
                if len(sp) != 2:
                    print(f"Malformed input: {inp}, Expected: Name, Error")
                    continue
            measurable_name, measurable_error = sp
            try:
                measurable_error: np.float64 = np.float64(measurable_error)
            except ValueError:
                print(f"Not a float: {measurable_error}")
                continue
            measurables_strings[measurable_name] = measurable_error

    measureables = Measureables()
    for measurable_name, measurable_error in measurables_strings.items():
        measurable_symbol = Symbol(measurable_name)
        measureables[(measurable_symbol, measurable_name)] = measurable_error

    return measureables


class RJustLine:
    def __init__(self):
        self.laststrlen = 0

    def __call__(self, string: str, n, fill=" "):
        r = (n - self.laststrlen) * fill + string
        self.laststrlen = len(string)
        return r


def parse_excel(measurements):
    return [[float(y) for y in x.strip().split("\t")] for x in measurements.strip().split("\n") if x.strip()]


@wrap_kwarg("measurables", Measureables.cast)
def evaluate_measurements(
    var: Symbol = None,
    expr: sympy_Expr = None,
    err_expr: sympy_Expr = None,
    partials=None,
    measurements: list[list[np.float64]] = None,
    measurement_measureables_mapping: callable[[list[np.float64]], dict[Symbol, np.float64]] = None,
    measurables: Measureables = None,
    decimals=4,
):
    print("Compiling expressions...", end="", flush=True)
    __start_time = time.time()
    numba_signature = nb.float64(*(nb.float64,) * len(measurables))
    flat_expr = flatten_expr(expr, partials, measurables)
    flat_err_expr = flatten_expr(err_expr, partials, measurables)
    flat_expr_lambda_compiled = nb.njit(numba_signature)(lambdify([x[1] for x in measurables.keys()], flat_expr, dummify=False, modules=["numpy"]))
    flat_err_expr_lambda_compiled = nb.njit(numba_signature)(lambdify([x[1] for x in measurables.keys()], flat_err_expr, dummify=False, modules=["numpy"]))
    __elapsed = time.time() - __start_time
    print(f" elapsed time: {round(__elapsed*1000, 1)}ms")
    vals = []
    errs = []
    print(f"{"Evaluating measurements"}          (computation time)")
    i = 1
    n_measurements = len(measurements)
    __elapsed_list = []
    for m in measurements:
        params: dict[str, np.float64] = {}
        for k, v in measurement_measureables_mapping(m).items():
            params[measurables.get_str_for_symbol_key(k)] = v
        __start_time = time.time()
        val = round(flat_expr_lambda_compiled(**params), decimals)
        err_val = round(flat_err_expr_lambda_compiled(**params), decimals)
        __elapsed = time.time() - __start_time
        vals.append(val)
        errs.append(err_val)
        s = f"{i}/{n_measurements}"
        __elapsed_list.append(__elapsed * 1000)
        print(f"#{s:8}    {val:.{decimals}f}±{err_val:.{decimals}f}       ({round(__elapsed*1000, 1)}ms)")
        i += 1
    val_mean = round(sum(vals) / len(vals), decimals)
    err_mean = round(sum(errs) / len(errs), decimals)
    rjust = RJustLine()
    rel_err = f"{round(100*err_mean/val_mean, 2):.2f}%"
    padd = " " * 10
    print(decimals * 2 + 5)
    print(f"{rjust("mean", len(str(var))+1)}  {rjust("rel. error", decimals*2+5)}{rjust("cumul. runtime", len(rel_err)+len(padd))}")
    print(f"{var}={val_mean:.{decimals}f}±{err_mean:.{decimals}f}  {rel_err}{padd}({round(sum(__elapsed_list), 1):.4f}ms)")


def do_everything(
    var: Symbol,
    expr: sympy_Expr,
    partials: dict[Symbol, sympy_Expr],
    measurables: dict[Symbol, float] | Measureables,
    measurements: list[list[np.float64]],
    measurement_measureables_mapping: callable[[list[np.float64]], dict[Symbol, np.float64]],
    outtex: Path = None,
    decimals=4,
):

    err_expr_formal, target_err_expr = do_error_calc(
        target_var=var,
        target_expr=expr,
        partials=partials,
        measurables=measurables,
        outtex=outtex,
    )

    evaluate_measurements(
        var=var,
        expr=expr,
        err_expr=target_err_expr,
        partials=partials,
        measurables=measurables,
        measurements=measurements,
        measurement_measureables_mapping=measurement_measureables_mapping,
        decimals=decimals,
    )


if __name__ == "__main__":
    # generate_measureables_tui()
    v, g, h, x, α_rad, t, x_coord, x_start, z_coord, z_start, α_deg = symbols("v g h x α_{rad} t x_{coord} x_{start} z_{coord} z_{start} α_{deg}")

    do_everything(
        var=v,
        expr=x * sqrt(g / (h * cos(2 * α_rad) + h + x * sin(2 * α_rad))),
        partials={x: sqrt((x_coord - x_start) ** 2 + (z_coord - z_start) ** 2), α_rad: α_deg * pi / 180},
        measurables={x_coord: 0.0005, x_start: 0.0005, z_coord: 0.0005, z_start: 0.0005, α_deg: 0.0005, h: 0, g: 0.0008},
        measurements=parse_excel(
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
        measurement_measureables_mapping=lambda m: {α_deg: m[0], x_coord: m[1], z_coord: m[2], x_start: 0.5, z_start: 0.5, h: 1.6, g: 0.239},
        outtex=None,
        decimals=4,
    )
    do_everything(
        var=g,
        expr=2 * (h + x * tan(α_rad)) / t**2,
        partials={x: sqrt((x_coord - x_start) ** 2 + (z_coord - z_start) ** 2), α_rad: α_deg * pi / 180},
        measurables={x_coord: 0.0005, x_start: 0.0005, z_coord: 0.0005, z_start: 0.0005, α_deg: 0.0005, t: 0.5, h: 0},
        measurements=parse_excel(
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
        measurement_measureables_mapping=lambda m: {α_deg: m[0], x_coord: m[1], z_coord: m[2], x_start: 0.5, z_start: 0.5, h: 1.6, t: m[3]},
        outtex=None,
        decimals=4,
    )
