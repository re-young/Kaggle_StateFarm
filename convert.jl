
# This script builds an HDF5 file out of the training data for
# State Farm Distracted Driver Detection


using HDF5
using Colors
using Images

searchdir(path,key) = filter(x->contains(x,key), readdir(path))

#Get all the file names.
#Folder denotes the image class
imgFiles=[]
labels=[]

for i=0:9
    files=searchdir("train/c$i/","jpg")
    for ff in files
        push!(imgFiles,"train/c$i/"*ff)
        push!(labels,i)
    end
end


#fix constats
n_data=200#size(imgFiles)[1]
tmp=load(imgFiles[1])
tmp=convert(Array{Gray},tmp)
w=size(tmp)[1]
h=size(tmp)[2]

#open hdf5 data objs

h5open("train_test.hdf5", "w") do h5
    dset_data = d_create(h5, "data", datatype(Float32), dataspace(w, h, 1, n_data))
    dset_label = d_create(h5, "label", datatype(Float32), dataspace(1, n_data))

	# for each image set load and store it in the hdf5 data object
	for i in 1:n_data

		if mod(i,250)==0
			println("Proccessing image number: ",i)
		end
	   
	    img=load(imgFiles[i]);

	    #convert to greyscale
	    img=convert(Array{Gray},img)
	    img=convert(Array{Float32},img)

	    dset_data[:,:,1,i] =img
	    dset_label[1,i] = convert(Float32,labels[i])
	end

end