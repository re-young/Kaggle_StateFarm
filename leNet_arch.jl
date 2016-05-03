# This impliments a nerual network with the LeNet Architecture
# The input data is convert to a HDF5 file using https://github.com/re-young/Kaggle_StateFarm/blob/master/convert.jl
# The conversion script could be edited to hold back a validation set

# TO DO:
# 	-upload results


using Mocha
srand(12345678)

# define the layers
data_layer  = AsyncHDF5DataLayer(name="data-train", source="train.txt", batch_size=32, shuffle=true)
conv_layer  = ConvolutionLayer(name="conv1", n_filter=20, kernel=(5,5), bottoms=[:data], tops=[:conv])
pool_layer  = PoolingLayer(name="pool1", kernel=(2,2), stride=(2,2), bottoms=[:conv], tops=[:pool])
conv2_layer = ConvolutionLayer(name="conv2", n_filter=50, kernel=(5,5), bottoms=[:pool], tops=[:conv2])
pool2_layer = PoolingLayer(name="pool2", kernel=(2,2), stride=(2,2), bottoms=[:conv2], tops=[:pool2])
fc1_layer   = InnerProductLayer(name="ip1", output_dim=500, neuron=Neurons.ReLU(), bottoms=[:pool2], tops=[:ip1])
fc2_layer   = InnerProductLayer(name="ip2", output_dim=10, bottoms=[:ip1], tops=[:ip2])
loss_layer  = SoftmaxLossLayer(name="loss", bottoms=[:ip2,:label])

backend = DefaultBackend()
init(backend)

common_layers = [conv_layer, pool_layer, conv2_layer, pool2_layer, fc1_layer, fc2_layer]
net = Net("LeNet_arch", backend, [data_layer, common_layers..., loss_layer])


exp_dir = "snapshots-$(Mocha.default_backend_type)"
method = SGD()
params = make_solver_parameters(method, max_iter=10000, regu_coef=0.0005,
                                mom_policy=MomPolicy.Fixed(0.9),
                                lr_policy=LRPolicy.Inv(0.01, 0.0001, 0.75),
                                load_from=exp_dir)
solver = Solver(method, params)


setup_coffee_lounge(solver, save_into="$exp_dir/statistics.jld", every_n_iter=1000)

# report training progress every 100 iterations
add_coffee_break(solver, TrainingSummary(), every_n_iter=100)

# save snapshots every 5000 iterations
add_coffee_break(solver, Snapshot(exp_dir), every_n_iter=5000)

# train the solver
solve(solver, net)

destroy(net)
destroy(test_net)
shutdown(backend)




