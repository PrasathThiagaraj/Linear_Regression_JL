using CSV,GLM,Plots,TypedTables

data=CSV.File("salary_dataset.csv")
X=data.YearsExperience
Y=data.Salary
gr(size=(600,600))
t = Table(X=X,Y=Y)
P_scatter=scatter(X,Y,
xlims=(0,20),
ylims=(0,122000),
xlabel="Years of Experience",
ylabel="salary",
title="salary_DataSet",
legend=false,
color=:red)

ols=lm(@formula(Y~X),t)
plot!(X,predict(ols),color=:black,linewidth=3)

newX= Table(X=[1250])
predict(ols,newX)

########################## #USING MACHINE LEARNING# ######################
epochs=0
theta_0=1
theta_1=4
h(x)=theta_0.+theta_1*X
plot!(X,h(X),color=:Blue,linewidth=3)
y_hat=h(X)
m=length(X)

####################### FUNCTIONS ##########################
function cost(X,Y)
    (1/2*m)*sum((y_hat-Y).^2)
end
J=cost(X,Y)
J_history=[]
push!(J_history,J)
function pd_theta_X0(X,Y)
    (1/m)*sum(y_hat-Y)
end
function pd_theta_X1(X,Y)
    (1/m)*sum((y_hat-Y).*X)
end

alpha_0=0.01
alpha_1=0.01
for i in 1:200

################# BEGIN ITERATIONS ############################
theta_0_temp = pd_theta_X0(X,Y)
theta_1_temp = pd_theta_X1(X,Y)

theta_0 -=alpha_0*theta_0_temp
theta_1 -=alpha_1*theta_1_temp 

y_hat=h(X)
J=cost(X,Y)
push!(J_history,J)
epochs+=1
display(plot!(X,y_hat,color=:green,alpha=0.5,
title="Salary_dataset(epochs=$epochs)"  
))
end
########################## END ITERATIONS #########################

plot!(X,y_hat,color=:Blue,alpha=0.5,linewidth=3,
title="Salary_dataset(epochs=$epochs)"  
)

# for measuring performance
plot!(X,predict(ols),color=:pink,linewidth=3)

# Learning curve
gr(size=(600,600))
p_Line= plot(0:epochs,J_history,
xlabel="Epochs",
ylabel="Cost",
title="Learning Curve",
color=:blue,
linewidth=2)