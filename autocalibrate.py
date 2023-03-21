# -------------------------------------------------------------------------------
# The confidential and proprietary information contained in this file may
# only be used by a person authorised under and to the extent permitted
# by a subsisting licensing agreement from Arm Limited or its affiliates.
#
#           (C) COPYRIGHT 2020-2023 Arm Limited or its affiliates.
#                  ALL RIGHTS RESERVED
# This entire notice must be reproduced on all copies of this file
# and copies of this file may only be made by a person if such person is
# permitted to do so under the terms of a subsisting license agreement
# from Arm Limited or its affiliates.
# -------------------------------------------------------------------------------
# Authors: Steffen Jensen, Salil Akerkar
#! /usr/bin/env python3

# Description:
# Usage:

import os, sys, shutil
import pandas as pd
from collections import OrderedDict
from collections import defaultdict

import json, re
import argparse
import common
import copy
import matplotlib.pyplot as plt
from common import *
from collections import OrderedDict
import statistics
from statistics import mode

#######################################################################
##
##  Globals
##
#######################################################################
InstInfo = {
    "a64": {
        "itype": ["CORE", "AdvSIMD", "CRYPTO"]
    },
    "manual": {},
    "sve2": {
        "itype": ["CORE", "AdvSIMD", "CRYPTO", "SVE"]
    },
}

_SVE, _SME2, _A64, _SIMD, _SVE2 = range(5)
_NanoSuiteMap = {
    "sve": _SVE,
    "sme2": _SME2,
    "a64": _A64,
    "simd": _SIMD,
    "sve2": _SVE2
}
_NanobenchInfo = {
    _SVE: {
        "suite": "sve",
        "tests": "nanos_sme2_ssve",
        "inst_file": "sve2"
    },
    _SME2: {
        "suite": "sme2",
        "tests": "nanos_sme2_sme",
        "inst_file": "mortlach"
    },
    _A64: {
        "suite": "a64",
        "tests": "nanos_a64_8_6_a64",
        "inst_file": "a64"
    },
    _SIMD: {
        "suite": "simd",
        "tests": "nanos_a64simd_8_6_a64",
        "inst_file": "a64"
    },
    _SVE2: {
        "suite": "sve2",
        "tests": "nanos_sve2_a64",
        "inst_file": "sve2"
    },
}
_NanobenchSuites = [_NanobenchInfo[k]["suite"] for k in _NanobenchInfo.keys()]
_NanobenchChoice = [suite + "_bw" for suite in _NanobenchSuites
                    ] + [suite + "_lat" for suite in _NanobenchSuites]
SuiteType = ["lat", "bw"]


#######################################################################
##
##  Exps:  Experiments to be performed
##
#######################################################################
class Exps:
    def __init__(self, vl, cpu):

        self.args = copy.deepcopy(
            common._CPUInfo[common._CPUType[cpu]]["args"])
        self.vl = int(vl)
        self.name = "%s_vl%d" % (cpu, vl)
        self.cpu = cpu

    def get_args(self):
        return " ".join(self.args)

    def is_io(self):
        return not common._CPUInfo[common._CPUType[self.cpu]]["o3"]

    def is_o3(self):
        return common._CPUInfo[common._CPUType[self.cpu]]["o3"]

    def get_usim_iq(self):
        return common._CPUInfo[common._CPUType[self.cpu]]["iq"]

    def get_vl_128(self):
        return self.vl

    def get_vl_64(self):
        return self.vl * 2

    def get_name(self):
        return self.name

    def get_cpu(self):
        return self.cpu

    def get_model(self):
        return common._CPUInfo[common._CPUType[self.cpu]]["model"]

    def print_me(self):
        print("{:32} {:}".format(self.get_name(), self.get_args()))


job_harness_uarchsim = """
#!/bin/bash

# Is mrun available?
if [ -e /arm/tools/setup/bin/mrun ]; then
    export LD_LIBRARY_PATH=/arm/devsys-tools/collections/gcc/7.3.0/1/linux_2.6-redhat_10.6-x86_64/gcc-4.4.7-SYSTEM/lib64:$LD_LIBRARY_PATH
    mruncmd="/arm/tools/setup/bin/mrun - +core +arm/license/validation"
else
    export ARMLMD_LICENSE_FILE=7010@euhpc-lic07.euhpc.arm.com:7010@euhpc-lic03.euhpc.arm.com:7010@euhpc-lic04.euhpc.arm.com:7010@euhpc-lic05.euhpc.arm.com:7010@nahpc-lic09.nahpc.arm.com:8224@blr-lic03.blr.arm.com:8224@blr-lic04.blr.arm.com:7010@nahpc-lic06.nahpc.arm.com:8224@blr-lic05.blr.arm.com
    mruncmd=""
fi

echo "Start time: $(date --rfc-3339=seconds)" > harness.timestamp.txt


uarchsim=_REPLACE_UARCHSIM_BASE
plugin_path=/projects/randd/shoji/FastModel/latest/PVModelLib/external/plugins/Linux64_GCC-7.3/
plugin_name=ScalableVectorExtension
smeplugin=/projects/randd/shoji/FastModel/sve-future/ShojiPlugin/sve-future/Linux64_GCC-7.3/${plugin_name}

#    --plugin /projects/randd/shoji/FastModel/latest/PVModelLib/external/plugins/Linux64_GCC-7.3/TarmacTrace.so \
#    --plugin ${plugin_path}/TarmacTrace.so \

model_base=/projects/randd/shoji/FastModel/sve-future/Models/models/Linux64_GCC-7.3
${mruncmd} ${model_base}/FVP_Base_AEMvA \\
    --cyclelimit 800000 \\
    --quiet \\
    -C bp.terminal_0.start_telnet=0 \\
    -C bp.vis.disable_visualisation=1 \\
    --plugin ${smeplugin}.so \\
    -C SVE.${plugin_name}.veclen=_REPLACE_VL_64 \\
    -C SVE.${plugin_name}.sme_veclens_implemented=4 \\
    -C bp.secure_memory=false \\
    -C bp.pl011_uart0.shutdown_on_eot=1 \\
    -C bp.pl011_uart0.out_file="_REPLACE_out_file" \\
    -C cluster0.NUM_CORES=1 \\
    -C cluster0.imp_def_functionality_behaviour=1 \\
    -C cluster0.has_arm_v8-1=1 \\
    -C cluster0.has_arm_v8-2=1 \\
    -C cluster0.has_arm_v8-3=1 \\
    -C cluster0.has_arm_v8-4=1 \\
    -C cluster0.has_arm_v8-5=1 \\
    -C cluster0.has_arm_v8-6=1 \\
    -C cluster0.has_arm_v8-7=1 \\
    --plugin ${uarchsim}/Big_Core.so \\
    -C TRACE.Big_Core.stats_on_pmu=1 -C TRACE.Big_Core.use_hints=1 \\
    -C TRACE.Big_Core.output=_REPLACE_simlog \\
    -C TRACE.Big_Core.stats=_REPLACE_stats \\
    -C TRACE.Big_Core.pipetrace=_REPLACE_pipetrace \\
    -C TRACE.Big_Core.branch_stats=_REPLACE_branch_stats \\
    -C TRACE.Big_Core.vl=_REPLACE_VL_64 \\
    -C TRACE.Big_Core.sme_vl=_REPLACE_VL_128 \\
    -C TRACE.Big_Core.interval=1000000 \\
    -C TRACE.Big_Core.oplist=_REPLACE_CFG_PATH/a64.instructions \\
    -C TRACE.Big_Core.macrolist=_REPLACE_CFG_PATH/a64.macros \\
    -C TRACE.Big_Core.ext_oplist=_REPLACE_CFG_PATH/sve2.instructions \\
    -C TRACE.Big_Core.ext_macrolist=_REPLACE_CFG_PATH/sve2.macros \\
    -C TRACE.Big_Core.extraoplist=_REPLACE_CFG_PATH/manual.instructions \\
    -C TRACE.Big_Core.extramacrolist=_REPLACE_CFG_PATH/manual.macros \\
    -C TRACE.Big_Core.options=_REPLACE_OVERRIDE_DAT \\
    -C TRACE.Big_Core.inlineopts="sme_fp_issue_queue__REPLACE_USIM_IQ=1; fp_issue_queue__REPLACE_USIM_IQ=1; disable_line_crossing_penalty=1; use_new_fp_timings = 1;" \\
    -C TRACE.Big_Core.graph_start=1 \\
    -C TRACE.Big_Core.graph_stop=100000000 \\
    _REPLACE_PROGRAM_PATH

retval=$?

echo "End time: $(date --rfc-3339=seconds)" >> harness.timestamp.txt

exit ${retval}
"""


#######################################################################
##
##  Function used to dispatch created jobs
##
#######################################################################
# create_jobs: function used for creating jobs
# inputs:
#   ti: struct containing metadata for all jobs to create
# returns:
#   doesnt return anything, but creates run scripts for each job along with a file structure at the specified location
def create_jobs(ti):
    njobs = 0
    job = job_harness_uarchsim.replace("_REPLACE_UARCHSIM_BASE",
                                       args.uarchsim_base)
    cfg_base = os.path.join(tools_base,
                            common._CPUInfo[common._CPUType[args.cpu]]["sym"])
    override_dat = os.path.join(cfg_base, "override.dat")
    for workload in ti.workloads:
        testjob = job.replace(
            "_REPLACE_PROGRAM_PATH",
            os.path.join(workload["where"], workload["exename"]),
        )
        testjob = testjob.replace("_REPLACE_OVERRIDE_DAT", override_dat)
        testjob = testjob.replace("_REPLACE_CFG_PATH", cfg_base)
        # print("workload is at " + workload["where"])
        # print("binary is called " + workload["exename"])
        # print("workload name =  " + workload["name"])

        for (name, exp) in ti.allExps.items():
            # print("name = " + name)

            testpath = os.path.join(workload["rundir"], exp.get_model(),
                                    workload["name"])
            # print('rundir = ' + workload['rundir'])
            (jobname, outname) = get_job_names(exp)

            exe_prefix = workload["exename"].split(".")[0]
            jobname = jobname.split(".")[0] + "_" + exe_prefix + ".sh"
            expjob = testjob.replace("_REPLACE_VL_64", "%s" % exp.get_vl_64())
            expjob = expjob.replace("_REPLACE_VL_128", "%s" % exp.get_vl_128())
            # expjob = expjob.replace("_REPLACE_START_ADDR", get_start_addr(workload['where']))
            expjob = expjob.replace("_REPLACE_USIM_IQ", exp.get_usim_iq())
            expjob = expjob.replace(
                "_REPLACE_out_file",
                "%s" % os.path.join(testpath, exe_prefix + ".log"))
            expjob = expjob.replace(
                "_REPLACE_simlog",
                "%s" % os.path.join(testpath, exe_prefix + "_simlog.out"),
            )
            expjob = expjob.replace(
                "_REPLACE_stats",
                "%s" % os.path.join(testpath, exe_prefix + "_stats.out"),
            )
            expjob = expjob.replace(
                "_REPLACE_pipetrace",
                "%s" % os.path.join(testpath, exe_prefix + "_pipetrace.txt"),
            )
            expjob = expjob.replace(
                "_REPLACE_branch_stats",
                "%s" % os.path.join(testpath,
                                    exe_prefix + "_branch_stats.dat.gz"),
            )
            # print("testpath = " + testpath)
            if not os.path.exists(testpath):
                os.makedirs(testpath)
            jobpath = os.path.join(testpath, jobname)

            # print("jobpath = " + jobpath)

            with open(jobpath, "w") as jfile:
                jfile.write(expjob)
            os.chmod(jobpath, 0o755)
            njobs += 1
    print("Creating %d Jobs" % (njobs))


#######################################################################
##
##  Classes and Functions to analyze the stats.txt and print the results
##
#######################################################################
def create_exps(ti):
    exp_str = ti.args.exp
    exps = dict()
    vl_opts = list()
    args = exp_str.split(";")
    for arg in args:
        (key, val) = arg.split(":")
        if key == "vl":
            vl_opts = val.split(",")
        elif key == "cpu":
            cpu_opts = val.split(",")
        else:
            assert 0, "Unexpected arg %s in args.exp" % (key, args.exp)
    for vl in vl_opts:
        _vl = int(vl)
        for _cpu in cpu_opts:
            e = Exps(_vl, _cpu)
            exps[e.get_name()] = e
    od = OrderedDict(sorted(exps.items()))
    return od


def crawl_workloads(ti):
    workloads = []
    root = ti.args.testdir
    # print("root = " + root)
    for dirpath, dummy, filenames in os.walk(root):
        for file in filenames:
            if "axf" not in file:
                continue
            if not filtered_test(ti.testFilter, file[:-4]):
                continue
            workload = dict()
            workload["where"] = dirpath
            workload["rundir"] = ti.args.rundir
            workload["exename"] = file
            workload["name"] = file[:-4]
            workloads.append(workload)
        break
    return workloads


# get_decoder_df: function which parses the decoder file and returns a dataframe containing jname along with corresponding opcodes and bit masks
# inputs:
#   decoder_json: path to decoder json file
#   output_file: path to output file to write if desired
#   ins_length: length of instruction, should pretty much always be 32
# returns:
#   df: dataframe containing jnames along with corresponding opcodes and bit masks
def get_decoder_df(decoder_json="./decoder.json",
                   output_file="",
                   ins_length=32):
    f = open(args.decoder_json)
    decoder = json.load(f)

    cols = ["jname", "decode", "mask"]
    jnames = []
    decodes = []
    masks = []
    # num = 0
    # jname_dup_num = 0
    for ins in decoder["instances"]:
        # if(ins["form_id"] in jnames):
        # print("found dup jname: " + ins["form_id"])
        # jname_dup_num += 1

        jnames.append(ins["form_id"])
        decode = ""
        mask = ""
        for i in range(ins_length - 1, -1, -1):
            if str(i) in ins["decode"]:
                decode += str(ins["decode"][str(i)])
                mask += "1"
            else:
                decode += "0"
                mask += "0"

        # print('decode = ' + decode)
        decodes.append(decode)
        masks.append(mask)

    data = list(zip(jnames, decodes, masks))
    df = pd.DataFrame(data, columns=cols)
    if output_file:
        df.to_csv(output_file)
    # print("jname_dup_num = " + str(jname_dup_num))
    # print("total_jnames = " +str(len(decoder["instances"])))
    # assert False
    return df


def add_braces_to_tbl_tbx(instr: str) -> str:
    instr = instr.strip()
    if not (instr.startswith("TBL") or instr.startswith("TBX")):
        return instr

    operands = instr.split()[1:]
    num_operands = len(operands)

    if num_operands >= 2 and "{" not in operands[1]:
        operands[1] = "{" + " " + operands[1]

    if num_operands > 3 and "}" not in operands[-3]:
        if "}" in operands[-2]:
            operands[-2] = operands[-2].replace("}", "")
        operands[-3] = operands[-3] + " " + "}"

    if num_operands == 3 and "}" not in operands[-2] and "{" not in operands[1]:
        operands[1] = "{" + " " + operands[1]
        operands[-2] = operands[-2] + " " + "}"

    instr = " ".join([instr.split()[0]] + operands)
    return instr


def update_instr(instr: str) -> str:
    instr = instr.strip()

     # Check if instruction starts with "TBL", "TBX", "EXT", or "SPLICE"
    if not any(instr.startswith(x) for x in ["TBL", "TBX", "EXT", "SPLICE"]):
        return instr



    # Split the operands of the instruction
    operands = instr.split()[1:]
    num_operands = len(operands)

    # Count the number of "V" or "Z" present in the instruction except for the first "V" or "Z"
    if instr.startswith("TBL") or instr.startswith("TBX"):
        var_prefix = "V"
    else:
        var_prefix = "Z"

    num_vars = 0
    for i in range(1, num_operands):
        if operands[i].startswith(var_prefix):
            num_vars += 1

    # Replace the second "V" or "Z" onwards with a sequence of "V" or "Z" starting from 0 to the total number of "V" or "Z" minus one
    var_counter = 0
    for i in range(1, num_operands):
        if operands[i].startswith(var_prefix):
            old_var_operand = operands[i]
            new_var_operand = var_prefix + str(
                var_counter) + old_var_operand[old_var_operand.index("."):]
            operands[i] = new_var_operand
            var_counter += 1
    if instr.startswith("TBL"):
        var_prefix = "Z"
    num_vars = 0
    for i in range(1, num_operands):
        if operands[i].startswith(var_prefix):
            num_vars += 1

    # Replace the second "V" or "Z" onwards with a sequence of "V" or "Z" starting from 0 to the total number of "V" or "Z" minus one
    var_counter = 0
    for i in range(1, num_operands):
        if operands[i].startswith(var_prefix):
            old_var_operand = operands[i]
            new_var_operand = var_prefix + str(
                var_counter) + old_var_operand[old_var_operand.index("."):]
            operands[i] = new_var_operand
            var_counter += 1
    # Combine the modified operands and join with the instruction to form the final instruction
    instr = " ".join([instr.split()[0]] + operands)
    return instr

def replace_tbl_tbx_mov(instr: str) -> str:
    instr = instr.replace("MOVPRFX", "MOV")
    instr = instr.replace("MOVS", "MOV")
    if instr.startswith("TBX Z0.H") or instr.startswith(
        "TBX Z0.S") or instr.startswith(
            "TBX Z0.D") or instr.startswith("TBX Z0.B"):
        instr = instr.replace("{", "")
        instr = instr.replace("}", "")
    if instr.startswith("TBL Z0.H") or instr.startswith(
        "TBL Z0.S") or instr.startswith(
            "TBLZ0.D") or instr.startswith("TBL Z0.B"):
        instr = instr.replace("{", "")
        instr = instr.replace("}", "")
    if instr.startswith("MOV Z0.H") or instr.startswith(
        "MOV Z0.S") or instr.startswith(
            "MOV Z0.D") or instr.startswith("MOV Z0.B"):
        instr = instr.replace("P0/Z", "P0/M")
    if instr.startswith("TBL Z0.H") or instr.startswith(
        "TBL Z0.S") or instr.startswith(
            "TBL Z0.D") or instr.startswith("TBL Z0.B"):
        operands = instr.split(",")
        if "{" not in operands[1]:
            operands[1] = "{" + operands[1].strip()
        if "}" not in operands[-2]:
            operands[-2] = operands[-2].strip() + "}"
        instr = ",".join(operands)
    if instr.startswith("MOV ") and instr.endswith("Z0, Z0"):
        instr = ""

    return instr

def replace_operands(new_op):
    new_op = new_op.replace("dn", "0")
    new_op = new_op.replace("d", "0")
    new_op = new_op.replace("u+0", "0")
    new_op = new_op.replace("p", "0")
    new_op = new_op.replace("0-1", "1")
    new_op = new_op.replace("0-2", "2")
    new_op = new_op.replace("0-3", "3")
    return new_op
# write_nanobench_asm_file: function which writes instructions in nanobench xls to an assemply file, also removes any bandwidth-suite instructions from the input dataframe
# inputs:
#   df: dataframe containing data from input nanobench xls file
#   suite: lat or bw
# outputs:
#   df: same as input but with bandwidth-suite instructions filtered out
#   Also creates nanobench.s which is an assembly file containing all instructions within nanobench
def write_nanobench_asm_file(df, suite="lat"):
    f = open("nanobench.s", "w")
    drop_ind_list = []

    for i in range(len(df["Form"])):
        if f"_{suite}_" in df["Test"][i]:
            new_op = df["Form"][i]
            new_op = replace_operands(new_op)
            new_op = add_braces_to_tbl_tbx(new_op)
            new_op = update_instr(new_op)
            new_op = replace_tbl_tbx_mov(new_op)

            f.write(new_op + "\n")
        else:
            drop_ind_list.append(i)

    df = df.drop(drop_ind_list)
    df.reset_index(drop=True, inplace=True)

    f.close()
    return df


# compile_nanobench_asm: function which adds nanobench forms to an assembly file and tries to compile, iteratively
# removing instructions which do not compile until all remaining instructions compile. Also removes non-compiling
# instructions from the input dataframe
# inputs:
#   nanobench_df: dataframe containing nanobench data from xls
# returns:
#   nanobench_df: same as input dataframe but without instructions which didnt compile
#   Also creates a file called a.out which contains the compiled binary of te assembl file which gets created
def compile_nanobench_asm(nanobench_df):
    err_detected = True
    err_line_nums = set([])
    undef_symbols = set([])
    error_lines = []

    while err_detected:
        print("==========Compiling nanobench.s==========")
        err_out = subprocess.Popen(
            [
                "/arm/warehouse/ARMCC/TestableTools/6.18/25/linux-x86_64-none-rel/bin/armclang",
                "--target=aarch64-arm-none-eabi",
                "-march=armv8.6-a+memtag+aes+crypto+bf16+fp16+fp16fml+sve+sve2+sve2-aes+sve2-bitperm+sve2-sha3+sve2-sm4",
                "nanobench.s",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).communicate()[1]
        err_detected = False
        err_lines = err_out.split(b"\n")

        for err_line in err_lines:
            # compile errors
            if re.match("nanobench.s:\d+:\d+: error: *", err_line.decode()):
                err_detected = True
                err_tokens = err_line.split(b":")
                err_line_nums.add(int(err_tokens[1]) - 1)
                error_lines.append(err_line.decode())
            elif re.match("Error: (.*): Undefined symbol (.*)",
                          err_line.decode()):
                err_detected = True
                err_tokens = err_line.split(b" ")
                undef_symbols.add(err_tokens[4].decode("utf-8").strip())
                error_lines.append(err_line.decode())

        if len(undef_symbols):
            f = open("nanobench.s", "r")
            nanobench_lines = f.readlines()
            f.close()
            for i in range(len(nanobench_lines)):
                line = nanobench_lines[i]
                if any(s in line for s in undef_symbols):
                    err_line_nums.add(i)

        if err_detected:
            nanobench_df = nanobench_df.drop(err_line_nums)
            nanobench_df.reset_index(drop=True, inplace=True)
            with open('error_file', 'w') as file:
                for error_line in error_lines:
                    file.write(error_line)
            break
            nanobench_df = nanobench_df.drop(err_line_nums)
            nanobench_df.reset_index(drop=True, inplace=True)

    return nanobench_df


# correlate_jname_ins: function which correlates the forms within nanobench_df to the jnames within the decoder
# inputs:
#   suite: suite name
#   nanobench_df: dataframe containing info from nanobench xls file
#   decoder: dataframe containing decoder jnames along with corresponding opcodes and masks
# returns:
#   out_dict: dict containing jnames correlated to each form along with rtl ipc and cpi
def correlate_jname_ins(suite, nanobench_df, decoder):
    f = open("a.txt", "r")
    lines = f.readlines()
    start = 1
    for line in lines:
        if "$x.0" in line:
            break
        start += 1

    out_dict = {}
    # not_found_num = 0
    # dup_num = 0
    suite_delim = "_" + suite + "_"
    print("==========Matching jnames With Instruction Binaries==========")
    for i in range(len(nanobench_df["Form"])):
        test_full_name = nanobench_df["Test"][i]
        if suite_delim not in test_full_name:
            continue
        # print("test_full_name = " + test_full_name)
        suite_name = (test_full_name.split(suite_delim)[0] + "_" +
                      suite).removeprefix("mb_monk_")
        test_name = test_full_name.split(suite_delim)[1]
        if suite_name not in out_dict.keys():
            out_dict[suite_name] = {}
        if test_name not in out_dict[suite_name].keys():
            out_dict[suite_name][test_name] = {}

        line = lines[start + i]
        # print('line ' + str(i) + ' = ' + line)
        if len(line.split()) < 2:
            continue  # skip this line
        inst_opcode = int(line.split()[1], 16)

        op = nanobench_df["Form"][i]
        ipc = nanobench_df["RTL"][i]
        cpi = 1.0 / ipc
        latency = max(1, round(cpi))
        out_dict[suite_name][test_name][op] = {}
        out_dict[suite_name][test_name][op]["jname"] = ""
        out_dict[suite_name][test_name][op]["IPC"] = ipc
        out_dict[suite_name][test_name][op]["CPI"] = cpi
        out_dict[suite_name][test_name][op]["latency"] = latency
        found_num = 0
        for j in range(len(decoder["jname"])):
            decode = decoder["decode"][j]
            # print('decode = ' + decode)
            opcode_mask = decoder["mask"][j]
            # print('opcode_mask = ' + opcode_mask)
            opcode_mask_int = int(opcode_mask, 2)
            # print('opcode_mask_int = ' + str(opcode_mask_int))
            masked_inst_opcode = inst_opcode & opcode_mask_int
            # print('masked_inst_opcdode = ' + str(masked_inst_opcode))
            masked_bin_string = format(masked_inst_opcode, "b").zfill(32)
            # print('masked_bin_string = ' + masked_bin_string)
            # print('unmasked_bin_string = ' + format(inst_opcode, 'b').zfill(32))
            if decode == masked_bin_string:
                out_dict[suite_name][test_name][op]["jname"] = decoder[
                    "jname"][j]
                # found_num += 1
        # if(not found_num):
        #    print("form not found: " + op)
        #    not_found_num += 1
        # elif(found_num > 1):
        #    print("form matches to multiple jnames: " + op)
        #    dup_num += 1

    # print("not_found_num = " + str(not_found_num))
    # print("dup_num = " + str(dup_num))
    # print("total_num = " + str(len(nanobench_df["Form"])))
    return out_dict


# process_nanobench_data: function which parses nanobench data from xls and correlates it to jnames from decoder file
# inputs:
#   decoder_json: path to the decoder file containing jnames along with opcodes
#   nanobench_xls: path to xls file containing nanobench data
#   output_json_path: path to location where output json file should be saved
#   suite: lat or bw
# returns:
#   out_dict: dict containing metadata for each form along with klein rtl results, also should print this info to a json file if output_json_path is specified
def process_nanobench_data(decoder_json,
                           nanobench_xls,
                           output_json_path="",
                           suite="lat"):
    decoder = get_decoder_df(decoder_json=decoder_json)  # get decoder info
    nanobench_df = pd.read_excel(
        nanobench_xls, skiprows=1)  # read nanobench info from xls sheet
    nanobench_df = write_nanobench_asm_file(nanobench_df,
                                            suite)  # write nanobench asm file
    nanobench_df = compile_nanobench_asm(
        nanobench_df)  # compile nanobench asm file

    os.system(
        "/arm/warehouse/ARMCC/TestableTools/6.18/25/linux-x86_64-none-rel/bin/fromelf --text -c a.out > a.txt"
    )  # decompile nanobench asm file in order to get instruction binaries
    out_dict = dict()
    out_json = open(output_json_path, "w")
    out = correlate_jname_ins(suite, nanobench_df,
                              decoder)  # correlate jnames to nanobench forms
    if output_json_path:  # makeoutput json if flag specified
        json.dump(out, out_json, indent=4)
    out_dict.update(out)
    out_json.close()
    return out_dict


# parse dependencies from obj dump: function which determines dependencies used for each form in obj dump file
# inputs:
#   obj_dump_path: path to obj_dump file for test
# returns:
#   dependencies: list of dependencies corresponding to each form in the test. RAW for Read-after-write, WAW for Write after write, X for not detected
def parse_dependencies_from_obj_dump(obj_dump_path):
    obj_dump_file = open(obj_dump_path, "r")
    obj_dump_lines = obj_dump_file.readlines()
    dependencies = []
    for i in range(len(obj_dump_lines)):
        obj_dump_line = obj_dump_lines[i]
        if "_loop>:" in obj_dump_line:
            ins1 = re.split(r"[\s,{}]+", obj_dump_lines[i + 10])
            ins2 = re.split(r"[\s,{}]+", obj_dump_lines[i + 11])
            ins1_tokens = ins1[4:-1]
            ins2_tokens = ins2[4:-1]
            # print("ins1 = " + str(ins1))
            # print("ins2 = " + str(ins2))
            # print("ins1_tokens = " + str(ins1_tokens))
            # print("ins2_tokens = " + str(ins2_tokens))

            match = -1
            if len(ins1_tokens) > 0:
                dest1 = ins1_tokens[0].split(".")[0]
                dest1 = dest1.split("/")[0]
                # dest1_is_wx_reg = (re.match("W\d+", dest1) or re.match("X\d+", dest1))
                for j in range(len(ins2_tokens)):
                    dest2 = ins2_tokens[j].split(".")[0]
                    dest2 = dest2.split("/")[0]
                    # dest2_is_wx_reg = (re.match("W\d+", dest2) or re.match("X\d+", dest2))
                    if dest1 == dest2:
                        match = j
                        break

            if match == 0:
                dependencies.append("WAW")
            elif match > 0:
                dependencies.append("RAW")
            else:
                dependencies.append("X")
    obj_dump_file.close()
    return dependencies


# parse_stats: function which parses ipc, cycle count, and instruction count from uarchsim stats.out file
# inputs:
#   stats_path: path to *_stats.out log file for uarchsim job
# returns:
#   ipc_list: list containing ipc for all forms in stats file
#   cycles_list: list containing cycle count for all forms in stats file
#   ins_list: list containing instruction count for all forms in stats file
def parse_stats(stats_path):
    stats_file = open(stats_path, "r")
    stats_lines = stats_file.readlines()
    ipc_list = []
    cycles_list = []
    ins_list = []
    for i in range(len(stats_lines)):
        stats_line = stats_lines[i]
        if "identifier:" in stats_line:
            ipc_list.append(float(stats_lines[i + 1].split(":")[1].strip()))
            cycles_list.append(int(stats_lines[i + 2].split(":")[1].strip()))
            ins_list.append(int(stats_lines[i + 3].split(":")[1].strip()))
    stats_file.close()
    return ipc_list, cycles_list, ins_list


# parse_uarchsim_log: function which parses uarchsim results for a particular job into a dict
# inputs:
#   log_path: path to uarchsim *.log output
#   obj_dump_path: path to obj_dump output
#   stats_path: path to *_stats.out output
# returns:
#   results_dict: dict containing results for uarchsim job
def parse_uarchsim_log(log_path, obj_dump_path, stats_path):
    dependencies = parse_dependencies_from_obj_dump(obj_dump_path)
    ipc_list, cycles_list, ins_list = parse_stats(stats_path)

    results_dict = OrderedDict()
    log_file = open(log_path, "r")
    log_lines = log_file.readlines()
    num = 0
    for i in range(len(log_lines)):
        log_line = log_lines[i]
        # print('log_line = ' + log_line)
        if "Starting test" in log_line:
            # print('form_line = ' + log_lines[i+2])
            form = log_lines[i + 2].split("=")[1].strip()
            results_dict[form] = {}
            results_dict[form]["IPC"] = ipc_list[num]
            results_dict[form]["CPI"] = 1.0 / ipc_list[num]
            results_dict[form]["cycles"] = cycles_list[num]
            results_dict[form]["instructions"] = ins_list[num]
            results_dict[form]["dependency"] = dependencies[num]
            num += 1
    # print("results dict = " + str(results_dict))
    return results_dict


# process_uarchsim_data: function which collects uarchsim results from log files and puts them in dictionary
# inputs:
#   ti: struct containing relevant metadata for all uarchsim jobs
# returns:
#   out_dict: dictionary containing results from each uarchsim job, also prints this info to a json file
def process_uarchsim_data(ti):
    # print('in process_uarchsim_data')
    out_dict = {}
    out_dict["meta"] = {}
    suite = ti.workloads[0]["rundir"].split("/")[-1].removeprefix("nanos_")
    out_dict["meta"]["suite"] = suite
    out_dict["meta"]["bench_type"] = out_dict["meta"]["suite"].split("_")[-1]
    out_dict["meta"]["model"] = "uarchsim"

    results_dict = {}
    for workload in ti.workloads:
        for (name, exp) in ti.allExps.items():
            testpath = os.path.join(workload["rundir"], exp.get_model(),
                                    workload["name"])
            log_name = workload["name"] + ".log"
            obj_dump_name = workload["name"] + ".txt"
            stats_name = workload["name"] + "_stats.out"
            log_path = os.path.join(testpath, log_name)
            obj_dump_path = os.path.join(testpath, obj_dump_name)
            stats_path = os.path.join(testpath, stats_name)
            # print("log_path = " + log_path)
            # print("obj_dump_path = " + obj_dump_path)
            # print("stats_path = " + stats_path)
            results_dict[workload["name"]] = parse_uarchsim_log(
                log_path, obj_dump_path, stats_path)

    out_dict["results"] = results_dict
    out_json = open((suite + "_results.json"), "w")
    json.dump(out_dict, out_json, indent=4)
    out_json.close()
    return out_dict


# plot_error_hist: function which makes plot of data from csv containing uarchsim and rtl results
# inputs:
#   final_df: dataframe containins uarchsim and rtl results and metadata
#   title: title to give histogram
#   width: desired width of histogram
#   height: desired height of histogram
#   filter_col: name of column in dataframe to filter based on if filtering is desired
#   filter_val: value which will be used to filter out data to be plotted.
#   e.g. if filter_col is "Dependency Type", and filter_val is "RAW", then only plot values in histogram with a value of "RAW" in the "Dependency Type" Column
# returns:
# doesnt return anything, btu produces histogram of relevant data
def plot_error_hist(final_df,
                    title="",
                    width=6.4,
                    height=4.8,
                    filter_col="",
                    filter_val=""):
    if filter_col and filter_val:
        pct_error_list = final_df.loc[final_df[filter_col] == filter_val][
            "CPI % Difference"]
    else:
        pct_error_list = final_df["CPI % Difference"]
    # bins = range(0, int(max(pct_error_list)) + bin_size, bin_size)
    bins = [0, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    max_err = max(pct_error_list)
    cutoff = len(bins)
    for i in range(len(bins)):
        if bins[i] > max_err:
            cutoff = i + 1
            break

    bins = bins[0:cutoff]

    plt.figure(figsize=(width, height))
    plt.hist(pct_error_list, bins)
    plt.title(title)
    plt.ylabel("Number of Forms")
    plt.xlabel("CPI % Error between RTL and Uarchsim")
    plt.xticks(bins[1:], rotation=45, fontsize=8)
    plt.xlim([0, max(bins)])
    plt.show()


def get_insts(inst_file):
    insts = defaultdict(list)
    f = open(inst_file, "r")
    lines = f.readlines()
    for line in lines:
        # print(line)
        if line.startswith("#===="):
            i = dict()
        elif line.startswith("#") or line == "\n":
            continue
        elif ":" in line:
            i[line.split(":")[0].strip()] = line.split(":")[1].strip()
        elif line.startswith("End"):
            # change to list instead of dict and lookup based on
            # some other logic
            insts[i["jname"]].append(i)
        else:
            assert False, "Unhandled line: %s" % line
    f.close()
    return insts


def get_insts_dup_num(inst_file):
    insts = dict()
    f = open(inst_file, "r")
    lines = f.readlines()
    dup_num = 0
    for line in lines:
        # print(line)
        if line.startswith("#===="):
            i = dict()
        elif line.startswith("#"):
            continue
        elif ":" in line:
            i[line.split(":")[0].strip()] = line.split(":")[1].strip()
        elif line.startswith("End"):
            # change to list instead of dict and lookup based on
            # some other logic
            key = i["jname"]
            if key in insts:
                print("Duplicate found: " + key)
                dup_num += 1
            else:
                insts[key] = i
        else:
            assert False, "Unhandled line: %s" % line
    f.close()
    return dup_num


def inst_filtered(arch_inst, suite):
    if suite == "manual":
        return False
    if arch_inst["status"] == "Red":
        return True
    if arch_inst["type"] not in InstInfo[suite]["itype"]:
        return True
    return False


def auto_update(latencies):
    cfg_base = os.path.join(tools_base,
                            common._CPUInfo[common._CPUType[args.cpu]]["sym"])
    override_dat = os.path.join(cfg_base, "override.dat")
    pipe_info = process_config(override_dat)
    file = (_NanobenchInfo[_NanoSuiteMap[args.nanobench.split("_")[0]]]
            ["inst_file"] + ".instructions")
    instruction_file = os.path.join(cfg_base, file)
    out_file = instruction_file.replace("instructions", "instructions.bak")
    name = os.path.basename(instruction_file).split(".")[0]
    insts = get_insts(instruction_file)
    create_inst_file(out_file, insts, pipe_info, latencies)


def create_inst_file(out, insts, pipe_info, latencies):
    f = open(out, "w")
    for (jname, l) in iter(insts.items()):
        found = False
        for inst in l:
            f.write("#==========================================\n")
            if jname in latencies.keys():
                found = True
                most_common = mode(latencies[jname])
                pipe_depth = pipe_info[inst["unit"]]

            for (key, val) in iter(inst.items()):
                if found and key == "latency":
                    if most_common <= pipe_depth:
                        val = "normal"
                    else:
                        val = most_common - pipe_depth
                f.write("%s: %s\n" % (key, val))
            f.write("End\n")
    f.close()


def write_insts(inst_class, out, insts, arch):
    f = open(out, "w")
    for (jname, l) in iter(insts.items()):
        if jname not in arch.keys():
            print("jname/form_id:%s not found in %s" % (jname,
                                                        args.decoder_json))
        elif inst_filtered(arch[jname], inst_class):
            continue
        for inst in l:
            f.write("#==========================================\n")
            for (key, val) in iter(inst.items()):
                f.write("%s: %s\n" % (key, val))
            f.write("End\n")
    f.close()


def validate_instruction_file(inst_class, inst_file, out_file):
    f = open(args.decoder_json)
    decoder = json.load(f)
    arch = dict()
    for ins in decoder["instances"]:
        arch[ins["form_id"]] = ins
    # print("inst_dup_num = " + str(get_insts_dup_num(inst_file)))
    insts = get_insts(inst_file)
    write_insts(inst_class, out_file, insts, arch)


def validate_instructions():
    cfg_base = os.path.join(tools_base,
                            common._CPUInfo[common._CPUType[args.cpu]]["sym"])
    for file in os.listdir(cfg_base):
        if file.endswith(".instructions"):
            instruction_file = os.path.join(cfg_base, file)
            out_file = instruction_file.replace("instructions",
                                                "instructions.bak")
            name = os.path.basename(instruction_file).split(".")[0]
            validate_instruction_file(name, instruction_file, out_file)


def process_config(override_dat):
    pipe_info = dict()
    f = open(override_dat, "r")
    lines = f.readlines()
    in_exe_blk = False
    for line in lines:
        if "Start: EXU" in line:
            in_exe_blk = True
            blk = dict()
            continue
        elif "End: EXU" in line:
            in_exe_blk = False
            units = [u.strip() for u in blk["sub_units"].split(",")]
            for unit in units:
                pipe_info[unit] = int(blk["nominal_pipeline_depth"])
        if in_exe_blk:
            blk[line.split("=")[0].strip()] = line.split("=")[1].strip()
    f.close()
    return pipe_info


#######################################################################
##
##  Function used to analyze results
##
#######################################################################
# analyze_jobs: function used for creating jobs
# returns:
#   creates results csv and also auto updates latency files based on that
def analyze_jobs():
    print("==========args.analyze detected==========")
    objdump_workloads(ti)
    uarchsim_results = process_uarchsim_data(ti)
    # out_json = open('test.json', "w")
    # json.dump(uarchsim_results, out_json, indent=4)
    # out_json.close()
    # assert False
    suite = uarchsim_results["meta"]["suite"]
    suite_list = []
    test_list = []
    form_list = []
    jname_list = []
    klein_ipc_list = []
    klein_cpi_list = []
    uarchsim_ipc_list = []
    uarchsim_cpi_list = []
    cpi_pct_diff_list = []
    dependency_list = []
    latencies = dict()
    # print("uarchsim results = " + str(uarchsim_results))
    passing = 0
    total = 0
    for test in uarchsim_results["results"]:
        for form in uarchsim_results["results"][test]:
            suite_list.append(suite)
            test_list.append(test)
            form_list.append(form)
            uarchsim_ipc_list.append(
                uarchsim_results["results"][test][form]["IPC"])
            uarchsim_cpi_list.append(
                uarchsim_results["results"][test][form]["CPI"])
            dependency_list.append(
                uarchsim_results["results"][test][form]["dependency"])
            jname = ""
            klein_ipc = -1
            klein_cpi = -1
            cpi_pct_diff = -1

            if suite not in nanobench_data:
                print("suite not found: " + suite)
            elif test not in nanobench_data[suite]:
                print("test not found: " + test)
            elif form not in nanobench_data[suite][test]:
                print("form not found: " + form)
            else:
                jname = nanobench_data[suite][test][form]["jname"]
                klein_ipc = nanobench_data[suite][test][form]["IPC"]
                klein_cpi = nanobench_data[suite][test][form]["CPI"]
                cpi_pct_diff = 100 * abs(
                    (uarchsim_results["results"][test][form]["CPI"] -
                     klein_cpi) / klein_cpi)
                if cpi_pct_diff <= 10:
                    passing += 1
                total += 1

            jname_list.append(jname)
            klein_ipc_list.append(klein_ipc)
            klein_cpi_list.append(klein_cpi)
            cpi_pct_diff_list.append(cpi_pct_diff)
            if jname not in latencies.keys():
                latencies[jname] = list()
            latencies[jname].append(int(round(klein_cpi)))

    print("[%d/%d]Passing Percentage: %.2f" % (passing, total,
                                               (passing / total) * 100))
    columns = [
        "suite",
        "test",
        "form",
        "jname",
        "Klein IPC",
        "Klein CPI",
        "uarchsim IPC",
        "uarchsim CPI",
        "CPI % Difference",
        "Dependency Type",
    ]
    final_data = list(
        zip(
            suite_list,
            test_list,
            form_list,
            jname_list,
            klein_ipc_list,
            klein_cpi_list,
            uarchsim_ipc_list,
            uarchsim_cpi_list,
            cpi_pct_diff_list,
            dependency_list,
        ))
    final_df = pd.DataFrame(final_data, columns=columns)
    print("writing csv file")
    # print(final_data)
    if args.sort in columns:
        final_df.sort_values(by=[args.sort], inplace=True)
    fname = "ins_info_" + args.nanobench
    date = datetime.datetime.now()
    fname += "_" + date.strftime("%m%d%y")
    fname += ".csv"
    final_df.to_csv(fname)

    # auto-generate new latency file
    auto_update(latencies)


if __name__ == "__main__":
    global args
    global jobs_finished
    parser = argparse.ArgumentParser()
    ti = TestInfo()
    tools_base = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--cpu",
        help="CPU to calibrate",
        choices=common._CPUChoice,
        default="klein")
    parser.add_argument(
        "--nanobench", help="Run nanobench", choices=_NanobenchChoice)
    parser.add_argument(
        "--nano-prefix", help="Add prefix for nanobench results", default="./")
    parser.add_argument(
        "--uarchsim-base",
        help="uarchsim base",
        # debug build
        # default="/arm/projectscratch/pd/ajanta/users/salake01/dev/modelling/uarchsim/build/obj/debug/sme2p1.00bet0/aem_plugin/bigcore/"
        # release build
        default=
        "/arm/projectscratch/pd/ajanta/users/salake01/dev/modelling/uarchsim/build/install/release/sme2p1.00bet0",
        # default="/projects/randd/shoji/FastModel/UarchSimPlugin/sve2-HEAD/latest/",
    )
    parser.add_argument("--testdir", help="Test Dir")
    parser.add_argument(
        "--cleanup", help="Clean up results", action="store_true")
    parser.add_argument(
        "--exp",
        help="Exps to use [smoke|full|smcu:<0|1>;vl:<vl1,vl2,..>;cpu:<t,m>]",
        default="vl:1;cpu:klein",
    )
    parser.add_argument(
        "--dryrun", help="Show Jobs do not run", action="store_true")
    parser.add_argument("--run", help="Run Jobs", action="store_true")
    parser.add_argument("--create", help="Create Jobs", action="store_true")
    parser.add_argument("--testfilter", help="Test filter")
    parser.add_argument("--rundir", help="Rundir (default: run_dir)")
    parser.add_argument("--resume", help="Resume Jobs", action="store_true")
    parser.add_argument("--analyze", help="Analyze Jobs", action="store_true")
    parser.add_argument(
        "--tarmac_gen",
        help="Dump tarmac-trace and debug log",
        action="store_true")
    parser.add_argument(
        "--get-inst-info",
        help="Create uarchsim inst file based on nanobench results",
        action="store_true",
    )
    parser.add_argument(
        "--suite",
        help="Suite type used with --get-inst-info",
        default="lat",
        choices=SuiteType,
    )
    parser.add_argument(
        "--decoder-json",
        help="path to the decoder json file",
        default=tools_base + "/../../core/decoders/sme2p1.00bet0/decoder.json",
        # default=tools_base + "/../../core/decoders/a64/decoder.json",
    )
    parser.add_argument(
        "--sort",
        help="what param to sort final df based on",
        default="",
    )
    parser.add_argument(
        "--plot",
        help="name of csv to plot data from",
        default="",
    )
    parser.add_argument(
        "--validate-instructions",
        help="validate instruction files",
        action="store_true",
    )

    args = parser.parse_args()
    if not args.nanobench:
        args.tarmac_gen = True
    jobs_finished = 0
    ti.args = args
    if args.nanobench:
        (nanosuite, nanotype) = (
            args.nanobench.split("_")[0],
            args.nanobench.split("_")[1],
        )
    if args.rundir is None:
        args.rundir = "run_dir"
    (rundir, _) = get_results_name(ti)
    args.rundir = os.path.abspath("%s/%s" % (args.rundir, rundir))
    # nanobench_base = '/arm/projectscratch/pd/raven/salake01/dev/nanobench/'
    nanobench_base = "/arm/projectscratch/pd/ajanta/users/rajbho01/nanobenches/"
    nanobench_results = (
        "/arm/projectscratch/pd/ajanta/users/salake01/dev/modelling/dash/dataset/"
    )

    if args.nanobench:
        suite = _NanoSuiteMap[nanosuite]
        dirname = "%s_%s" % (_NanobenchInfo[suite]["tests"], nanotype)
        args.testdir = "%s/%s" % (nanobench_base, dirname)
        args.rundir += "/%s" % dirname
    if args.testfilter:
        ti.testFilter = args.testfilter.strip().split(":")

    nanobench_data = {}
    if args.get_inst_info:
        intermediate_nanobench_json = os.path.join(
            tools_base, "data", "nanobench_inst_%s_info.json" % args.suite)
        nanobench_data = process_nanobench_data(
            decoder_json=args.decoder_json,
            nanobench_xls=os.path.join(
                tools_base, "data",
                common._CPUInfo[common._CPUType[args.cpu]]["golden"]),
            output_json_path=intermediate_nanobench_json,
            suite=args.suite,
        )
        sys.exit()
    else:
        intermediate_nanobench_json = os.path.join(
            tools_base, "data", "nanobench_inst_%s_info.json" % nanotype)
        f = open(intermediate_nanobench_json)
        nanobench_data = json.load(f)
        f.close()

    ti.allExps = create_exps(ti)
    if args.testdir:
        ti.workloads = crawl_workloads(ti)

    for (name, exp) in ti.allExps.items():
        exp.print_me()

    if args.cleanup:
        cleanup(ti.workloads)
    if args.create:
        create_jobs(ti)
    if args.run:
        # better to create the jobs in case they
        # are not there already.
        if not args.create:
            create_jobs(ti)
        run_jobs(ti)
    if args.analyze:
        analyze_jobs()

    if args.plot:
        plot_data = pd.read_csv(args.plot)
        plot_error_hist(
            plot_data,
            title="Error Histogram, " + args.nanobench,
            width=8,
            height=5.8)

        # plot_error_hist(
        #    plot_data, title="Error Histogram, " + args.nanobench + ", WAW", width=8, height=5.8, filter_col="Dependency Type", filter_val="WAW"
        # )
    if args.validate_instructions:
        validate_instructions()