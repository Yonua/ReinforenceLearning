# 多臂赌博机:3臂

using Random
using Distributions
using Plots

mutable struct Mba
    q::Array{Float32}
    reward::Array{Float32}
    action::Int
    actionCounts::Array{Int}
    rewardUpdateProcessA::Array{Float32}
    rewardUpdateProcessB::Array{Float32}
    rewardUpdateProcessC::Array{Float32}
end

function choose_action(mba::Mba,policy::String,epsilon=0.2,c_ratio=0.5)
    action =1
    if policy == "mba"
        if Random.rand() < epsilon
            action = rand((1,2,3))
        else
            action = argmax(mba.q)
        end
    elseif policy == "ucb"
        if in(0,mba.actionCounts)
            action = findall(x->x==0,mba.actionCounts)[1]
        else
            value = mba.q .+ c_ratio*sqrt.(log(sum(mba.actionCounts))./mba.actionCounts)
            action = argmax(value)
        end
    elseif policy == "fix"
        action = argmax(mba.q)
    end
    return action
end

function take_reward(action::Int)
    r = 0.0
    if action == 1
        r = rand(Normal(1,1))
    elseif action == 2
        r = rand(Normal(2,1))
    elseif action == 3
        r = rand(Normal(1.5,1))
    end
    return r
end

mba = Mba([0.0,0.0,0.0],[0.0,0.0,0.0],0,[0,0,0],[],[],[])

for i = 1:2000
    action = choose_action(mba,"mba")
    reward = take_reward(action)
    mba.q[action] = (mba.q[action]*mba.actionCounts[action]+reward)/(mba.actionCounts[action]+1)
    mba.actionCounts[action] += 1
    push!(mba.reward,mba.reward[end]+reward)
    push!(mba.rewardUpdateProcessA,mba.q[1])
    push!(mba.rewardUpdateProcessB,mba.q[2])
    push!(mba.rewardUpdateProcessC,mba.q[3])
end

# plot(mba.reward,title = "Mba && Ucb reward")
plot(mba.rewardUpdateProcessA,title = "rewardUpdateProcess")
plot!(mba.rewardUpdateProcessB,title = "rewardUpdateProcess")
plot!(mba.rewardUpdateProcessC,title = "rewardUpdateProcess")
# mba.action = 0
# mba.q = [0.0,0.0,0.0]
# mba.reward = [0.0]
# mba.actionCounts = [0,0,0]
# mba.rewardUpdateProcessA = []
# mba.rewardUpdateProcessB = []
# mba.rewardUpdateProcessC = []

# for i = 1:2000
#     action = choose_action(mba,"ucb")
#     reward = take_reward(action)
#     mba.q[action] = (mba.q[action]*mba.actionCounts[action]+reward)/(mba.actionCounts[action]+1)
#     mba.actionCounts[action] += 1
#     push!(mba.reward,mba.reward[end]+reward)
# end

# plot!(mba.reward,title = "Mba && Ucb reward")

# mba.action = 0
# mba.q = [0.0,0.0,0.0]
# mba.reward = [0.0]
# mba.actionCounts = [0,0,0]
# mba.rewardUpdateProcessA = []
# mba.rewardUpdateProcessB = []
# mba.rewardUpdateProcessC = []

# for i = 1:2000
#     action = choose_action(mba,"fix")
#     reward = take_reward(action)
#     mba.q[action] = (mba.q[action]*mba.actionCounts[action]+reward)/(mba.actionCounts[action]+1)
#     mba.actionCounts[action] += 1
#     push!(mba.reward,mba.reward[end]+reward)
# end

# plot!(mba.reward,title = "Mba && Ucb reward")
