using CSV, GLM, Plots, TypedTables
# Load the dataset
#theta_0 - intercept
#theta_1 - slope
#h(x) - mx+b
#m - no of samples
#y_bar - predicted value of Y
#cost - cost function
#we use partial derivative (pd) , pd_theta_1 & pd_theta_0 are used to evaluate the second part of the 
#formula of the partial derivative and then alpha_1 and alpha_0 should be multiplied and then should be subtracted from theta_1 and theta_0

data = CSV.File("Normalized_Salary_dataset.csv")
X = data.YearsExperience
Y = data.Salary

# Plotting
gr(size = (600,600))
t = Table(X=X,Y=Y)
pr_scatter = scatter(X,Y,xlims=(0,1),ylims = (0,1),
xlabel = "Yrs of Exp", ylabel = "Salary", color = :red)
ols=lm(@formula(Y~X),t)
plot!(X,predict(ols),color=:green,linewidth=3)
#begin
# Initializing parameters
epochs = 0
theta_0 = 0
theta_1 = 0
#theta_0 = -0.013244159701219045
#theta_1 = 1.038921802479055

# Hypothesis function
h(x) = theta_0 .+ theta_1 .* x

# Number of samples
m = length(X)

# Cost function
function cost(X, Y, theta_0, theta_1)
    y_bar = h(X)
    return (1/(2*m)) * sum((y_bar - Y).^2)
end

# Partial derivatives
function pd_theta_0(X, Y, theta_0, theta_1)
    y_bar = h(X)
    return (1/m) * sum(y_bar .- Y)
end

function pd_theta_1(X, Y, theta_0, theta_1)
    y_bar = h(X)
    return (1/m) * sum((y_bar .- Y) .* X)
end

# Learning rates
alpha = 0.01

# Gradient Descent Loop
J_History = []

for i in 1:10000
    # Compute the gradients
    grad_theta_0 = pd_theta_0(X, Y, theta_0, theta_1)
    grad_theta_1 = pd_theta_1(X, Y, theta_0, theta_1)

    # Update the parameters
    theta_0 -= alpha * grad_theta_0
    theta_1 -= alpha * grad_theta_1

    # Compute the cost with updated parameters
    J = cost(X, Y, theta_0, theta_1)
    push!(J_History, J)
    
    # Plotting the current regression line
    y_bar = h(X)
    #display(plot!(X, y_bar, color = :blue, alpha = 0.5))
    
    # Increment epoch
    epochs += 1
end
#end
# Final plot
plot!(X, h(X), color = :blue, alpha = 0.5,linewidth = 3)
display(pr_scatter)
