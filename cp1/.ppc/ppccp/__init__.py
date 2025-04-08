import os
from typing import Optional
from ppcgrader.compiler import Compiler
import ppcgrader.config


class Config(ppcgrader.config.Config):
    def __init__(self,
                 single_precision: bool,
                 code: str,
                 openmp: bool = False,
                 gpu: bool = False,
                 vectorize: bool = True):
        from . import info
        super().__init__(binary='cp',
                         cfg_file=__file__,
                         info=info,
                         gpu=gpu,
                         openmp=openmp,
                         code=code)
        self.single_precision = single_precision
        self.vectorize = vectorize
        self.default_demo = [os.path.join(self.base_dir, 'demo.png')]
        self.demo_flags = self._demo_flags_png
        self.demo_post = self._demo_post_png

    def common_flags(self, compiler: Compiler) -> Compiler:
        compiler = super().common_flags(compiler)

        if self.single_precision:
            compiler = compiler.add_definition("PPC_ALLOW_FLOAT", 1)
        else:
            compiler = compiler.add_definition("PPC_ALLOW_FLOAT", 0)

        if not self.vectorize:
            compiler = compiler.add_flag("-fno-tree-vectorize")

        return compiler

    def parse_output(self, output):
        input_data = {
            "nx": None,
            "ny": None,
            "data": None,
        }
        output_data = {
            "result": None,
        }
        output_errors = {
            "max_error": None,
            "locations": None,
        }
        statistics = {}

        def parse_matrix(string):
            M = string.strip('[]').split(';')
            M = [row.strip() for row in M]
            M = [row.split(" ") for row in M]
            M = [[float(e) for e in row] for row in M]
            return M

        for line in output.splitlines():
            splitted = line.split('\t')
            if splitted[0] == 'result':
                errors = {
                    'fail': True,
                    'pass': False,
                    'done': False
                }[splitted[1]]
            elif splitted[0] == 'time':
                time = float(splitted[1])
            elif splitted[0] == 'perf_wall_clock_ns':
                time = int(splitted[1]) / 1e9
                statistics[splitted[0]] = int(splitted[1])
            elif splitted[0].startswith('perf_'):
                statistics[splitted[0]] = int(splitted[1])
            elif splitted[0] in [
                    'max_error', 'max_error_limit', 'gvfa_error',
                    'gvfa_error_limit'
            ]:
                output_errors[splitted[0]] = float(splitted[1])
            elif splitted[0] in ['ny', 'nx']:
                input_data[splitted[0]] = int(splitted[1])
            elif splitted[0] == 'input':
                input_data["data"] = parse_matrix(splitted[1])
            elif splitted[0] == 'output':
                output_data["result"] = parse_matrix(splitted[1])
            elif splitted[0] == 'locations':
                output_errors["locations"] = parse_matrix(splitted[1])

        nx = input_data.get('nx', None)
        ny = input_data.get('ny', None)
        if nx and ny and self.code != "cp9a":
            flops = ny * ny * nx + 4 * ny * nx
            statistics['operations'] = flops
            statistics['operations_name'] = "useful arithmetic operation"

        return time, errors, input_data, output_data, output_errors, statistics
