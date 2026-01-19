#utils/NOE_Engine.py
# ---------------------------------------------------------------------
# Copyright 2024 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""
This is the script of cix noe umd api for inference over npu.
"""

from libnoe import *
import numpy as np
import struct
import time
from typing import Union


def get_data_info(d_type : noe_data_type_t) -> tuple:
    """
    Get data type information based on the input data type identifier.

    Args:
        d_type (noe_data_type_t): The data type identifier to retrieve information for.

    Returns:
        tuple: A tuple containing:
            - numpy data type (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.float16),
            - minimum value of the data type,
            - maximum value of the data type,
            - numpy data type (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.float16)
    """
    if d_type == noe_data_type_t.NOE_DATA_TYPE_S8:
        type_info = (np.int8, -128, 127,np.int8)
    elif d_type == noe_data_type_t.NOE_DATA_TYPE_U8:
        type_info = (np.uint8, 0, 255,np.uint8)
    elif d_type == noe_data_type_t.NOE_DATA_TYPE_S16:
        type_info = (np.int16, -32768, 32767,np.int16)
    elif d_type == noe_data_type_t.NOE_DATA_TYPE_U16:
        type_info = (np.uint16, 0, 65535,np.uint16)
    elif d_type == noe_data_type_t.NOE_DATA_TYPE_S32:
        type_info = (np.int32, -2147483648, 2147483647,np.int32)
    elif d_type == noe_data_type_t.NOE_DATA_TYPE_U32:
        type_info = (np.uint32, 0, 4294967295,np.uint32)
    elif d_type == noe_data_type_t.NOE_DATA_TYPE_F16:
        type_info = (np.float16, 0, 0,np.float16)
    else:
        raise NotImplementedError(f"Not Implement d_type {d_type}")
    return type_info

class EngineInfer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.job_cfg = noe_create_job_cfg_t()

        self.npu = NPU()

        self.input_type = []
        self.input_dtype_min = []
        self.input_dtype_max = []
        self.intype = []
        self.in_tensor_desc = []

        self.output_type = []
        self.output_dtype_min = []
        self.output_dtype_max = []
        self.outtype = []
        self.out_tensor_desc = []

        # forward time infer vars
        self._acc_time = 0  # accumulate during time
        self._cnt_time = 0  # count times
        self._dur_time = 0  # during time of infer once
        self._dur_time_list = []

        # throughput infer vars
        self.start_time = 0
        self.dur_time = 0
        self.cnt_time = 0
        self.acc_time = 0
        self._thp_list = []  # throughout info list

        self._init_context()
        self._load_graph()
        self._setup_tensors(NOE_TENSOR_TYPE_INPUT)
        self._setup_tensors(NOE_TENSOR_TYPE_OUTPUT)
        self._create_job()

    def _init_context(self):
        if self.npu.noe_init_context() != 0:
            raise RuntimeError("npu: noe_init_context fail")
        print("npu: noe_init_context success")

    def _load_graph(self):
        """
        Load the graph from the model path into the NPU.

        """
        ret,graph_id = self.npu.noe_load_graph(self.model_path)
        if ret != 0:
            raise RuntimeError("npu: noe_load_graph failed")
        self.graph_id = graph_id
        print("npu: noe_load_graph success")

    def _get_tensor_count(self, tensor_type : int) -> int:
        """
        Retrieve the number of tensors of a specified type from the NPU associated with the current graph.

        Args:
            tensor_type (int): The type of tensor to query (e.g., input or output tensor type).

        Returns:
            int: The number of tensors of the specified type.

        """
        ret,count = self.npu.noe_get_tensor_count(self.graph_id, tensor_type)
        if ret != 0:
            raise RuntimeError(
                f"npu: noe_get_output_tensor failed for type {tensor_type}"
            )
        return count

    def _setup_tensors(self, tensor_type : int):
        """
        Set up the tensor descriptors and corresponding properties for the input or output tensors.

        Args:
            tensor_type (int): The type of tensor to set up (NOE_TENSOR_TYPE_INPUT or NOE_TENSOR_TYPE_OUTPUT).

        Raises:
            NotImplementedError: If the tensor_type is not recognized.
        """
        tensor_count = self._get_tensor_count(tensor_type)
        tensor_list = (
            self.in_tensor_desc
            if tensor_type == NOE_TENSOR_TYPE_INPUT
            else self.out_tensor_desc
        )
        tensor_properties = (
            (self.input_type, self.input_dtype_min, self.input_dtype_max, self.intype)
            if tensor_type == NOE_TENSOR_TYPE_INPUT
            else (
                self.output_type,
                self.output_dtype_min,
                self.output_dtype_max,
                self.outtype,
            )
        )

        for idx in range(tensor_count):
            desc = self.npu.noe_get_tensor_descriptor(self.graph_id, tensor_type, idx)
            tensor_list.append(desc)
            data_type_info = get_data_info(desc.data_type)
            for prop, value in zip(tensor_properties, data_type_info):
                prop.append(value)

        print(
            f"{'Input' if tensor_type == NOE_TENSOR_TYPE_INPUT else 'Output'} tensor count is {tensor_count}."
        )

    def _create_job(self):
        """
        Create a job for the NPU using the graph id and configuration.

        """
        ret,job_id = self.npu.noe_create_job(
            self.graph_id, self.job_cfg
        )
        if ret != 0:
            raise RuntimeError("npu: noe_create_job failed")
        self.job_id = job_id
        print("npu: noe_create_job success")

    def forward(self, input_datas : Union[list, np.ndarray]) -> list:
        """
        Perform a forward pass through the model using the provided input data.

        Args:
            input_datas (Union[list, np.ndarray]): A list of input data arrays or a single input array.
                                                The length of the input must match the expected input tensor descriptions.

        Returns:
            list: A list of output data after processing through the model, scaled and adjusted.

        Raises:
            AssertionError: If the input data type is not a list or the lengths do not match expected input tensor descriptions.
            RuntimeError: If retrieval of output tensor fails during inference.
        """
        if not isinstance(input_datas, list):
            input_datas = [input_datas]
        assert type(input_datas) == list, "input datas must a list."
        assert len(input_datas) == len(
            self.in_tensor_desc
        ), f"len of input_datas:{len(input_datas)} does not match expected: {len(self.in_tensor_desc)}."

        job_id = self.job_id
        self.output = []

        for i, input_data in enumerate(input_datas):
            input_data = np.round(
                input_data.astype(float) * self.in_tensor_desc[i].scale
                - self.in_tensor_desc[i].zero_point
            )
            input_data = np.clip(
                input_data, self.input_dtype_min[i], self.input_dtype_max[i]
            ).astype(self.input_type[i])
            assert (
                len(input_data.tobytes()) == self.in_tensor_desc[i].size
            ), f"the input size is {len(input_data.tobytes())}, must equal to {self.in_tensor_desc[i].size}"
            self.npu.noe_load_tensor(job_id, i, input_data.tobytes())

        # infer forward delay
        start_time = time.perf_counter()
        self.npu.noe_job_infer_sync(job_id, -1)
        end_time = time.perf_counter()
        self._dur_time = end_time - start_time
        self._dur_time_list.append(self._dur_time)
        self._acc_time += self._dur_time
        self._cnt_time += 1

        for j in range(len(self.out_tensor_desc)):
            retmap,data = self.npu.noe_get_tensor(
                job_id, NOE_TENSOR_TYPE_OUTPUT, j)
            if retmap != 0:
                raise RuntimeError("npu: noe_get_tensor failed")
            output_data = np.frombuffer(data, dtype=self.output_type[j])
            self.output.append(
                (output_data.astype(np.float32) + self.out_tensor_desc[j].zero_point)
                / self.out_tensor_desc[j].scale
            )

        return self.output

    def clean(self):
        """
        Clean up resources used by the NPU by performing the following tasks:
        1. Clean job.
        2. Unload graph.
        3. Deinitialize context.

        """
        ret = self.npu.noe_clean_job(self.job_id)
        if ret == 0:
            print("npu: noe_clean_job success")
        else:
            print("npu: noe_clean_job fail")
            exit(-1)
        ret = self.npu.noe_unload_graph(self.graph_id)
        if ret == 0:
            print("npu: noe_unload_graph success")
        else:
            print("npu: noe_unload_graph fail")
            exit(-1)
        ret = self.npu.noe_deinit_context()
        if ret == 0:
            print("npu: noe_deinit_context success")
        else:
            print("npu: noe_deinit_context fail")
            exit(-1)

    # forward delay time for once
    def get_cur_dur(self):
        return self._dur_time

    # forward fps time for once
    def get_cur_fps(self):
        return 1 / self._dur_time

    def fps_info(self):
        fps = self._cnt_time / self._acc_time
        return fps

    # avergae fps
    def get_ave_fps(self, batch_size=1):
        fps = self._cnt_time * batch_size / self._acc_time
        return fps

    def get_max_fps(self, batch_size=1):
        fps = batch_size / min(self._dur_time_list)
        return fps

    def time_start(self):
        self.start_time = time.perf_counter()

    def time_dur(self):
        self.dur_time = time.perf_counter() - self.start_time

    def get_thp(self, batch_size=1):
        self.time_dur()
        _thp = batch_size * 1 / self.dur_time
        self._thp_list.append(_thp)
        return _thp

    def get_max_thp(self):
        return max(self._thp_list)

    def get_ave_thp(self):
        return sum(self._thp_list) / len(self._thp_list)
