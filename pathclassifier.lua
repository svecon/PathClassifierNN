require 'torch'
require 'nn'
require 'optim'
require 'cunn'
print('ready')

saveDataTest = 'pathclass.data.test.ten'
saveDataTrain = 'pathclass.data.train.ten'

train_file = 'path.class.train.ascii.dat'
test_file  = 'path.class.test.ascii.dat'

cmd = torch.CmdLine()
cmd:text('Training a network')
cmd:text()
cmd:text('Options')
-- cmd:option('-seed',123,'initial random seed')
cmd:option('-loadnn',false,'boolean option')
cmd:option('-loaddata',false,'boolean option')
cmd:option('-savedata',false,'boolean option')
cmd:option('-weights',false,'boolean option')
cmd:option('-firstcat',false,'boolean option')
cmd:option('-secondcat',false,'boolean option')
cmd:option('-train',6000,'number of train data')
cmd:option('-test',1000,'number of test data')
cmd:option('-batch',250,'batch size')
cmd:option('-epochs',10,'epochs count')
cmd:option('-etest',1,'etest')
cmd:option('-onlytest',false,'only test')
-- cmd:option('-stroption','mystring','string option')
cmd:text()

-- parse input params
params = cmd:parse(arg)

paramstr = cmd:string('', params, {})


saveNNPath = 'pathclass.nn.dat' .. paramstr
saveNNPathErr = 'pathclass.nnbest.dat' .. paramstr

----------------------------------------------------------------------
print('==> loading dataset')

-- We load the dataset from disk, it's straightforward

maxPathLen = 275 -- 1520
alphabetSize = 128 - 32 -- not using first 32 bytes

function loadData(pathName, dataLen)
    local dataset = torch.Tensor(dataLen, maxPathLen, alphabetSize):type('torch.ByteTensor')
    local datasetParams = torch.Tensor(dataLen):type('torch.ShortTensor')
    local datasetcounter = 1
    local file = io.open(pathName)

    if not file then
        print('ERROR READING FILE ' .. pathName)
        os.exit()
    end

    for line in file:lines() do

        if datasetcounter > dataLen then break end

        local path, testvalue = unpack(line:split('", '))
        local path = path:sub(3, path:len()):gsub("\\\\", "/")
        local testvalue = tonumber(testvalue:sub(1, testvalue:len() - 2)) + 1

        if params.firstcat and testvalue == 3 then
            print('changing value to 1')
            testvalue = 1
        elseif params.secondcat and testvalue == 3 then
            print('changing value to 2')
            testvalue = 2
        end

        -- print(path, testvalue)

        for i = 1, #path do
            local charpos = path:byte(i) - 31
            if charpos > alphabetSize then charpos = alphabetSize end -- overflow discard

            -- print(datasetcounter, i, charpos)
            dataset[datasetcounter][i][charpos] = 1
            datasetParams[datasetcounter] = testvalue
        end

        -- print(dataset[datasetcounter], datasetParams[datasetcounter])

        datasetcounter = datasetcounter + 1
    end

    return {data = dataset, labels = datasetParams, n = math.min(dataLen, datasetcounter)}
end

function toHotTensor(cold)
    hot = torch.Tensor(maxPathLen, alphabetSize):type('torch.ByteTensor')

    for i=1,maxPathLen do
        if cold[i] == 0 then break end
        hot[i][cold[i]] = 1
    end

    return hot
end

function hotBatch(data, from, to)
    hotBatch = torch.Tensor(to - from + 1, maxPathLen, alphabetSize):type('torch.ByteTensor')

    for i=from,to do
        hotBatch[i] = toHotTensor(data[i])
    end
    return hotBatch
end

if not params.loaddata then
    print('Parsing data')
    train_data = loadData(train_file, params.train) -- 1859846
    test_data = loadData(test_file, params.test) -- 207407
else
    print('Loading data tensors')
    train_data = torch.load(saveDataTrain)
    test_data = torch.load(saveDataTest)
end

if params.savedata then
    print('Saving parsed data')
    torch.save(saveDataTrain, train_data)
    torch.save(saveDataTest, test_data)
    print('Saved')
end

print('Train data have ', train_data.n, 'data')
print('Test data have ', test_data.n, 'data')

train_data.data = train_data.data:cuda()
train_data.labels = train_data.labels:cuda()
test_data.data = test_data.data:cuda()
test_data.labels = test_data.labels:cuda()

if not params.loadnn then
    print("Initializing NEW NN")
    -- network structure taken from
    -- http://nbviewer.ipython.org/github/eladhoffer/Talks/blob/master/DL_class2015/Deep%20Learning%20with%20Torch.ipynb
    net = nn.Sequential()
    stringHeight = maxPathLen
    -- 280x128
    -- 352x128
    net:add(nn.TemporalConvolution(alphabetSize, 256, 7))
    net:add(nn.TemporalMaxPooling(3, 3))
    stringHeight = math.floor((stringHeight-7+1)/3)
    -- 115x256
    net:add(nn.TemporalConvolution(256, 256, 3))
    net:add(nn.TemporalMaxPooling(3, 3))
    stringHeight = math.floor((stringHeight-3+1)/3)
    -- (115-3+1)x256 = 113x256
    -- (113/3)x256 = 37x256
    net:add(nn.Reshape(stringHeight*256))
    net:add(nn.Linear(stringHeight*256,512))

    net:add(nn.SoftSign())
    net:add(nn.Linear(512,3))
                              -- ReLU activation function
    -- net:add(nn.SpatialConvolution(8, 16, 3, 3))
    -- net:add(nn.SpatialMaxPooling(2,2,2,2))
    -- net:add(nn.ReLU())
    -- net:add(nn.SpatialConvolution(16, 32, 3, 3))
    -- net:add(nn.View(32*4*4):setNumInputDims(3)) -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*4*4
    -- net:add(nn.Linear(32*4*4, 128))             -- fully connected layer (matrix multiplication between input and weights)
    -- net:add(nn.ReLU())
    -- net:add(nn.Dropout(0.5))                    --Dropout layer with p=0.5
                     -- 10 is the number of outputs of the network (in this case, 10 digits)

    -- loss is cross-entropy (== logsoftmax + classnllcriterion)
    print(net)

    -- move entire network to the GPU
    net = net:cuda()

    -- random initialization according to Xavier
    local tmp = math.sqrt(1. / net:get(1).bias:size(1))
    net:get(1).weight:uniform(-tmp, tmp)
    net:get(1).bias:zero()

    tmp = math.sqrt(1. / net:get(3).bias:size(1))
    net:get(3).weight:uniform(-tmp, tmp)
    net:get(3).bias:zero()

    tmp = math.sqrt(1. / net:get(6).bias:size(1))
    net:get(6).weight:uniform(-tmp, tmp)
    net:get(6).bias:zero()

    tmp = math.sqrt(1. / net:get(8).bias:size(1))
    net:get(8).weight:uniform(-tmp, tmp)
    net:get(8).bias:zero()
    print('Random init complete')
else
    print("Loading NN")
    net = torch.load(saveNNPath):cuda()
end

if params.weights then
    if params.firstcat or params.secondcat then
        print('lossweight to 1 10 0')
        lossweight = torch.Tensor({1, 10, 0})
    else
        print('lossweight to 1 8 12')
        lossweight = torch.Tensor({1, 8, 12})
    end
end
loss = nn.CrossEntropyCriterion(lossweight):cuda()
parameters, gradParameters = net:getParameters()
print(parameters:nElement())
-- initialize state for the optimizer
opt_state = {}

-- optimization parameters
nepochs = params.epochs
etest = params.etest
batch_size = params.batch

-- SGD
--opt_param = {
--    learningRate = 0.01,
--    momentum = 0.9,
--    learningRateDecay = 1e-3
--}

-- AdaDelta
execute_optimizer = optim.adadelta
opt_params = {
    rho = 0.9,
    eps = 1e-6
}

if path.exists(saveNNPathErr) then
    bestPercentage = torch.load(saveNNPathErr)
    print("Loaded best successrate ", bestPercentage)
else
    bestPercentage = 0
    print("Default best successrate 0")
end

print("Training NN...")
-- train the network
for e=1,nepochs do

    local timer = torch.Timer()
    local confmat = optim.ConfusionMatrix(3, {'0','1','2'})
    local train_err = 0

    local inputs = torch.Tensor(batch_size, maxPathLen, alphabetSize):cuda()
    local targets = torch.Tensor(batch_size):cuda()

    for b=1,train_data.n/batch_size do

        if params.onlytest then break end

        function batch_eval(x)

            local err = 0

            if x ~= parameters then
                parameters:copy(x)
            end

            -- DO NOT FORGET TO ZERO THE GRADIENT STORAGE
            -- BEFORE EACH MINIBATCH
            gradParameters:zero()

           -- fill up the batch vector
            for i=1,batch_size do
                local ndx = (b-1) * batch_size + i
                inputs[i]:copy(train_data.data[ndx])
                targets[i] = train_data.labels[ndx]
            end

            local ys = net:forward(inputs)
            local batch_err = loss:forward(ys, targets)
            local dt_dy = loss:backward(ys, targets)
            net:backward(inputs, dt_dy)

            -- add all results into the confusion matrix
            -- for i=1,batch_size do
            --     confmat:add(ys[i],targets[i])
            -- end
            confmat:batchAdd(ys,targets)

            train_err = train_err + batch_err
            return batch_err, gradParameters
        end

        execute_optimizer(batch_eval, parameters, opt_params, opt_state)
    end

    print('******************* EPOCH ' .. e .. ' ************************')

    print('TRAINING [error = ' .. train_err .. ']')
    print(confmat:__tostring__())

    -- compute testing error every etest epochs
    local terr = 0
    local p = 0
    if e % etest == 0 then
        confmat:zero()
        for i=1,test_data.n do
            local x = test_data.data[i]
            local t = test_data.labels[i]
            local y = net:forward(x)
            confmat:add(y, t)
            terr = terr + loss:forward(y, t)
        end
        -- local ys = net:forward(test_data.data)
        -- confmat:batchAdd(ys, test_data.labels)
        -- terr = terr + loss:forward(ys, test_data.labels)

        print('TESTING [error = ' .. terr .. ']')
        print(confmat:__tostring__())
    end

    print('Epoch took ' .. timer:time().real .. ' seconds.')

    if not params.onlytest and confmat.totalValid > bestPercentage then
        print('Saving NN with success:', confmat.totalValid)
        torch.save(saveNNPath, net)
        torch.save(saveNNPathErr, confmat.totalValid)
        bestPercentage = confmat.totalValid
        print('Saved')
    end
end

-- Let us pass the image through the first convolutional layer and look at the results
-- imgf = net:get(1):forward(img)
-- print('Source input')
-- itorch.image(img)
-- print('After convolution')
-- itorch.image(image.scale(image.toDisplayTensor({input=imgf}),400))
-- print('Convolutional filters')
-- scaled_weights = image.scale(image.toDisplayTensor({input=net:get(1).weight,padding=2}),300)
-- itorch.image(scaled_weights)
