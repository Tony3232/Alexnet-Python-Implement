from math import floor

num_classes = 1000

conv1_input_length = 227
conv1_input_thickness = 3

conv1_kernel_num = 96
conv1_kernel_length = 11
conv1_stride = 4
conv1_padding = 0

pool_stride = 2
pool_kernel_length = 3

conv1_output_length = floor((conv1_input_length-conv1_kernel_length)\
                          /conv1_stride+1)
conv1_output_thickness = conv1_kernel_num
conv1_parameters = conv1_kernel_length**2 * conv1_input_thickness \
                   * conv1_kernel_num + conv1_kernel_num

print("===============================================")
print(f"conv1 has {conv1_kernel_num} \
kernels {conv1_kernel_length} * {conv1_kernel_length} \
* {conv1_input_thickness} with stride {conv1_stride}")
print(f"conv1 input size: {conv1_input_length} * \
{conv1_input_length} * {conv1_input_thickness}")
print(f"conv1 output size: {conv1_output_length} * \
{conv1_output_length} * {conv1_output_thickness}")
print(f"conv1 parameters = {conv1_kernel_num} * \
{conv1_kernel_length} * {conv1_kernel_length} \
* {conv1_input_thickness} + {conv1_kernel_num} = {conv1_parameters}")
print("===============================================")

conv2_input_length = floor((conv1_output_length+2*conv1_padding-\
                            pool_kernel_length)/pool_stride + 1)
conv2_input_thickness = conv1_output_thickness
conv2_kernel_num = 256
conv2_kernel_length = 5
conv2_stride = 1
conv2_padding = 2
conv2_output_length = floor((conv2_input_length+2*conv2_padding\
                             -conv2_kernel_length)/conv2_stride+1)
conv2_output_thickness = conv2_kernel_num
conv2_parameters = conv2_kernel_length**2 * conv2_input_thickness \
                   * conv2_kernel_num + conv2_kernel_num

print(f"conv2 has {conv2_kernel_num} \
kernels {conv2_kernel_length} * {conv2_kernel_length} \
* {conv2_input_thickness} with stride {conv2_stride}")
print(f"conv2 input size: {conv2_input_length} * \
{conv2_input_length} * {conv2_input_thickness}")

print(f"conv2 output size: {conv2_output_length} * \
{conv2_output_length} * {conv2_output_thickness}")
print(f"conv2 parameters = {conv2_kernel_num} * \
{conv2_kernel_length} * {conv2_kernel_length} \
* {conv2_input_thickness} + {conv2_kernel_num} = {conv2_parameters}")
print("===============================================")


conv3_input_length = floor((conv2_output_length-pool_kernel_length)\
                           /pool_stride + 1)
conv3_input_thickness = conv2_output_thickness
conv3_kernel_num = 384
conv3_kernel_length = 3
conv3_stride = 1
conv3_padding = 1
conv3_output_length = floor((conv3_input_length+2*conv3_padding\
                             -conv3_kernel_length)/conv3_stride+1)
conv3_output_thickness = conv3_kernel_num
conv3_parameters = conv3_kernel_length**2 * conv3_input_thickness \
                   * conv3_kernel_num + conv3_kernel_num

print(f"conv3 has {conv3_kernel_num} \
kernels {conv3_kernel_length} * {conv3_kernel_length} \
* {conv3_input_thickness} with stride {conv3_stride}")
print(f"conv3 input size: {conv3_input_length} * \
{conv3_input_length} * {conv3_input_thickness}")
print(f"conv3 output size: {conv3_output_length} * \
{conv3_output_length} * {conv3_output_thickness}")
print(f"conv3 parameters = {conv3_kernel_num} * \
{conv3_kernel_length} * {conv3_kernel_length} \
* {conv3_input_thickness} + {conv3_kernel_num} = {conv3_parameters}")
print("===============================================")


conv4_input_length = conv3_input_length
conv4_input_thickness = conv3_output_thickness
conv4_kernel_num = 384
conv4_kernel_length = 3
conv4_stride = 1
conv4_padding = 1
conv4_output_length = floor((conv4_input_length+2*conv4_padding\
                             -conv4_kernel_length)/conv4_stride+1)
conv4_output_thickness = conv4_kernel_num
conv4_parameters = conv4_kernel_length**2 * conv4_input_thickness \
                   * conv4_kernel_num + conv4_kernel_num

print(f"conv4 has {conv4_kernel_num} \
kernels {conv4_kernel_length} * {conv4_kernel_length} \
* {conv4_input_thickness} with stride {conv4_stride}")
print(f"conv4 input size: {conv4_input_length} * \
{conv4_input_length} * {conv4_input_thickness}")
print(f"conv4 output size: {conv4_output_length} * \
{conv4_output_length} * {conv4_output_thickness}")
print(f"conv4 parameters = {conv4_kernel_num} * \
{conv4_kernel_length} * {conv4_kernel_length} \
* {conv4_input_thickness} + {conv4_kernel_num} = {conv4_parameters}")
print("===============================================")

conv5_input_length = conv4_input_length
conv5_input_thickness = conv4_output_thickness
conv5_kernel_num = 256
conv5_kernel_length = 3
conv5_stride = 1
conv5_padding = 1
conv5_output_length = floor((conv5_input_length+2*conv5_padding\
                             -conv5_kernel_length)/conv5_stride+1)
conv5_output_thickness = conv5_kernel_num
conv5_parameters = conv5_kernel_length**2 * conv5_input_thickness \
                   * conv5_kernel_num + conv5_kernel_num

print(f"conv5 has {conv5_kernel_num} \
kernels {conv5_kernel_length} * {conv5_kernel_length} \
* {conv5_input_thickness} with stride {conv5_stride}")
print(f"conv5 input size: {conv5_input_length} * \
{conv5_input_length} * {conv5_input_thickness}")
print(f"conv5 output size: {conv5_output_length} * \
{conv5_output_length} * {conv5_output_thickness}")
print(f"conv5 parameters = {conv5_kernel_num} * \
{conv5_kernel_length} * {conv5_kernel_length} \
* {conv5_input_thickness} + {conv5_kernel_num} = {conv5_parameters}")
print("===============================================")


fc1_input_length = floor((conv5_output_length-pool_kernel_length)\
                           /pool_stride + 1)
fc1_output_length = 4096
fc1_parameters = fc1_input_length**2 * conv5_output_thickness\
* fc1_output_length + fc1_output_length
print(f"fc1 input size: {fc1_input_length} * \
{fc1_input_length} * {conv5_output_thickness}")
print(f"fc1 output size: {fc1_output_length}")
print(f"fc1 parameters = {fc1_input_length} * \
{fc1_input_length} * {conv5_output_thickness} \
* {fc1_output_length} + {fc1_output_length} = {fc1_parameters }")
print("===============================================")

fc2_input_length = fc1_output_length
fc2_output_length = fc2_input_length
fc2_parameters = fc2_input_length * fc2_output_length + fc2_output_length 
print(f"fc2 input size: {fc2_input_length}")
print(f"fc2 output size: {fc2_output_length}")
print(f"fc2 parameters = {fc2_input_length} * \
{fc2_output_length} + {fc2_output_length} = {fc2_parameters}")
print("===============================================")

fc3_input_length = fc2_output_length
fc3_output_length = num_classes
fc3_parameters = fc3_input_length * fc3_output_length + fc3_output_length
print(f"fc3 input size: {fc3_input_length}")
print(f"fc3 output size: {fc3_output_length}")
print(f"fc3 parameters = {fc3_input_length} * \
{fc3_output_length} + {fc3_output_length} = {fc3_parameters}")
print("===============================================")
total_parameters = conv1_parameters + conv2_parameters\
                   + conv3_parameters + conv4_parameters\
                   + conv5_parameters + fc1_parameters \
                   + fc2_parameters + fc3_parameters 
print(f"total parameters = {total_parameters}")
print("===============================================")




