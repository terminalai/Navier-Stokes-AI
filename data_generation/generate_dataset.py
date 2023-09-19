import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append('/mnt/c/users/jedli/onedrive - nus high school/documents/computing studies/navier_stonks')
sys.path.append(r'C:\\Users\\admin\\navier\\Navier-Stokes-AI')

import tensorflow as tf

from utils import BoundaryCondition
from data_generation import burgers_equation, fisher_kpp_equation, zpk_equation

for equation in ["burgers", "fisher_kpp", "zpk"]:
    for bc_name in ["periodic", "dirichlet"]:
        print(f"Solving {equation} with boundary condition {bc_name}...")

        if equation == "burgers": equation_generator = burgers_equation
        elif equation == "fisher-kpp": equation_generator = fisher_kpp_equation
        elif equation == "zpk": equation_generator = zpk_equation

        if bc_name == "periodic": bc = BoundaryCondition.PERIODIC
        elif bc_name == "dirichlet": bc = BoundaryCondition.DIRICHLET
        elif bc_name == "neumann": bc = BoundaryCondition.NEUMANN

        serialized_features_dataset = tf.data.Dataset.from_generator(
            lambda: equation_generator(bc), output_types=tf.string, output_shapes=()
        )

        filename = f'../data/{bc_name}_{equation}.tfrecord'
        writer = tf.data.experimental.TFRecordWriter(filename)
        writer.write(serialized_features_dataset)
