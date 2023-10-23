#!/usr/bin/env python
# coding: utf-8

# # AN INTRODUCTORY EXAMPLE



#%%


#%%




# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme();
import statsmodels.formula.api as sm

#from sklearn.preprocessing import scale
#import sklearn.linear_model as skl_lm
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf
import statistics as st
#from scipy import linalg


# In[ ]:





# **SCENARIO**: You are working with the marketing department of a car seller. Our boss asked you to create a report on sales and identify possible areas of development. We have the excel file provided by IT department. 

# First I load all the excel file directly in a table

# In[97]:


# Load CSV file on local
sales_table = pd.read_csv('/Users/giando/Courses/Python for Economic Applications/Data/sales_data.csv')

url = 'https://raw.githubusercontent.com/pal-dev-labs/Python-for-Economic-Applications/main/Data/sales_data.csv'
#sales_table = pd.read_csv(url)


# In[98]:


sales_table.info()


# In[90]:


sales_table


# I want to know how many different models each Manifacturer has

# In[100]:


sales_table['Manufacturer'].value_counts().plot.bar();


# I want to save this image for my future report. I need a nicer figure

# In[101]:


sales_table['Manufacturer'].value_counts().plot.bar()
plt.xlabel('Manufacturers')
plt.ylabel('Number of Models')
plt.title('Manufacturers different models')
plt.legend()
plt.savefig('manufacturer.png')  # this saves the figure i


# # TABLE MANIPULATION

# I want to extract total amount of sales for each manufacturer

# In[102]:


total_sales = pd.pivot_table(sales_table, index=['Manufacturer'], values=['Sales_in_thousands'],aggfunc=[np.sum])
total_sales


# Let's order a little bit

# In[103]:


total_sales = total_sales.sort_values(by=('sum', 'Sales_in_thousands'), ascending=False)
total_sales


# In[105]:


total_sales.plot.bar(legend = False)
plt.ylabel('Thousands of $');


# In[16]:


total_sales.iloc[0:15].plot.pie(subplots=True, legend= False);


# In[186]:


"""
column = sales_table['Sales_in_thousands'].values
list1 = []
for i in range(len(column)):
    list1.append(int(column[i]*100000 + np.random.randn()*8000000))
newcol = np.array(list1)
newcol = np.abs(newcol)
sales_table.insert(6,'TV Advert', newcol)
list2 = []
for i in range(len(column)):
    list2.append(int(((column[i])**2) + np.random.randn()*50000))
newcol = np.array(list2)
newcol = np.abs(newcol)
sales_table.insert(7,'Social Advert', newcol)
sales_table.to_csv('sales_data.csv')
"""


# ## I would like to understand if there are factors that influence the sales

# In[120]:


plt.scatter(sales_table['Price_in_thousands'].values, sales_table['Sales_in_thousands'].values)
plt.xlabel("Price_in_thousands");plt.ylabel("Sales_in_thousands");


# In[116]:


plt.scatter(sales_table['Fuel_efficiency'].values, sales_table['Sales_in_thousands'].values)
plt.xlabel("Engine_size");plt.ylabel("Sales_in_thousands");


# ## Let's try with more features

# In[118]:


sns.pairplot(sales_table.iloc[:,[3,4,5,6,7,8]]);


# ## Price, TV Advertising (very correlated) and Social Advertising seems interesting

# ## Let's try to calculate a correlation 

# In[135]:


cor_tv = sales_table['Sales_in_thousands'].corr(sales_table['TV Advert (thousands)'])
cor_social = sales_table['Sales_in_thousands'].corr(sales_table['Social Advert'])
cor_price = sales_table['Sales_in_thousands'].corr(sales_table['Price_in_thousands'])
print("Correlation between Sales and TV Advertising:", cor_tv)
print("Correlation between Sales and Social Advertising:", cor_social)
print("Correlation between Sales and Price:", cor_price)


# In[148]:


table1 = sales_table[['Sales_in_thousands','TV Advert (thousands)']]
table1 = table1.rename(columns={'TV Advert (thousands)': 'TV'})

#model = sm.ols('Sales_in_thousands ~ TV Advert (thousands)', data=sales_table)


# In[149]:


table1


# In[144]:


type(table1)


# In[172]:


import statsmodels.formula.api as sm

# Specify the linear regression model
model = sm.ols('Sales_in_thousands ~ TV', data=table1)

# Fit the model to the data
results = model.fit()

# Print the model summary
print(results.summary())


# In[177]:


model.predict


# In[ ]:





# In[152]:


from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense


# In[186]:


# Split the data into features and target
X = sales_table[['Fuel_efficiency', 'Power_perf_factor']]
y = sales_table['Awarded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[185]:


y


# In[188]:


# Create a sequential model
model = Sequential()

# Add a dense layer with 10 neurons and the relu activation function
model.add(Dense(10, activation='relu', input_shape=(2,)))

# Add a dense layer with 1 neuron and the sigmoid activation function
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[189]:


# Train the model on the training set
model.fit(X_train, y_train, epochs=100);


# In[190]:


# Evaluate the model on the testing set
loss, accuracy = model.evaluate(X_test, y_test)

# Print the loss and accuracy
print('Loss:', loss)
print('Accuracy:', accuracy)


# In[192]:


# Make predictions on new data
new_data = np.array([[1, 2]])
predictions = model.predict(new_data)

# Print the predictions
print('Predictions:', predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Everything Is an Object
# Python is an **object-oriented programming language (OOPP)** and in Python everything is an object.
# 
# In object-oriented programming languages like Python, an object is an entity that contains data along with associated metadata and/or functionality. 
# 
# In Python **everything is an object**, which means every entity has some metadata (called **attributes** or **fields**) and associated functionality (called **methods**). These attributes and methods are accessed via the dot syntax.
# 
# Example of objects are the integers numbers **1,2,3** or the symbols **"a"**,**"B"**,**"?"**. A "bigger" object could be a containers of numbers **[1,2,3,4...,9,..,101]**
# 
# As objects are so fundamental, Python has already some built-in objects, like numbers and characters

# In[ ]:





# In[ ]:





# In[ ]:





# Let's create an object **a**

# In[9]:


"a"


# We use the characters "" to create the object **'a'** whose data is the symbol *a*

# Let's create an object that is a number

# In[11]:


3


# the object **3** contains the symbol 3 (that python considers as the mathematical value 3

# Consider now

# In[14]:


"3"


# The object **'3'** is different from the object **3**. The first is the character *3* the second is the mathematical value 3

# Different objects behave differently when we apply operations

# In[18]:


3+3


# In[22]:


"3"+"3"


# Objects contain not only data or information but also **fields** and **methods**. We can use **dot** notation to access fields and methods
# 
# For example **capitalize** is a method to capitalize a symbol contained in a character object. It **produces** a new object that is the capitalized character
# 

# In[29]:


# capitalize is a method to capitalize the symbol a contained in the object 'a'. It produces the new object 'A'
"a".capitalize()


# In[ ]:





# In[2]:


"A".lower()


# The following picture summarize the objects creation process. We can note that every objects has a **TYPE** (or we can say it belongs to a **CLASS**)
# 
# <img src="fig/objects1.png">
# 

# In[ ]:





# # Types
# Objects have type information attached. 
# 
# There are built-in simple types offered by Python and several compound types, which will be discussed in the following lessons.
# 
# Python's simple types are summarized in the following table:
# 
# <center>**Python Scalar Types**</center>
# 
# | Type        | Example        | Description                                                  |
# |-------------|----------------|--------------------------------------------------------------|
# | ``int``     | ``x = 1``      | integers (i.e., whole numbers)                               |
# | ``float``   | ``x = 1.0``    | floating-point numbers (i.e., real numbers)                  |
# | ``complex`` | ``x = 1 + 2j`` | Complex numbers (i.e., numbers with real and imaginary part) |
# | ``bool``    | ``x = True``   | Boolean: True/False values                                   |
# | ``str``     | ``x = 'abc'``  | String: characters or text                                   |
# | ``NoneType``| ``x = None``   | Special object indicating nulls                              |
# 
# We'll take a quick look at each of these in turn.

# In[3]:


type(3)


# In[4]:


type("A")


# In[5]:


type(1.5)


# In[ ]:





# ## Python Variables
# Let's see how python manages variables.
# 
# We're going to assign the int value *10* to a variable named *a*, the str value "Stefano" to a variable named *name1* and a float *12.8* value to a variable named *c*

# In[7]:


# assignment instructions

a = 10     # assign 10 to variable a 
name1 = "Stefano"
c = 12.8


# In[ ]:





# In[ ]:





# ## Python Variables Are Pointers
# 
# Assigning variables in Python is as easy as putting a variable name to the left of the equals (``=``) sign:
# 
# ```python
# # assign 4 to the variable x
# a = 10
# ```
# 
# It seems as we create a space in memory, named a, and insert directly the value 10 in that space.
# 
# This is not the way in which Python works.
# 
# In Python variables are best thought of not as containers but as **pointers**.
# So in Python, when you write
# 
# ```python
# a = 10
# ```
# 
# you are essentially defining a *pointer* named ``a`` that points to an object in memory that contains the value ``10``. The right part of the assignment instruction above, creates an int object in memory and assignes the address memory of that object to the pointer ``a``. 
# 
# 
# In this way, variable a is able to accesso all the information of the object, including value, fields and methods.
# 

# In[21]:


a = "corso python tor vergata"
a


# In[22]:


type(a)


# In[34]:


a.capitalize()


# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


a.split()


# In[ ]:





# 
# Note one consequence of this: because Python variables just point to various objects, there is no need to "declare" the variable, or even require the variable to always point to information of the same type!
# This is the sense in which people say Python is *dynamically-typed*: variable names can point to objects of any type.
# So in Python, you can do things like this:

# In[23]:


x = 1         # x is an integer
x = 'hello'   # now x is a string


# There is a consequence of this "variable as pointer" approach that you need to be aware of. If we have two variable names pointing to the same mutable object, then changing one will change the other as well! For example, let's create and modify a list:

# In[25]:


x = [1, 2, 3]
y = x


# We've created two variables ``x`` and ``y`` which both point to the same object.
# Because of this, if we modify the list via one of its names, we'll see that the "other" list will be modified as well:

# In[ ]:





# In[27]:


print(y)


# In[28]:


x.append(4) # append 4 to the list pointed to by x
print(y) # y's list is modified as well!


# This behavior might seem confusing if you're wrongly thinking of variables as buckets that contain data.
# But if you're correctly thinking of variables as pointers to objects, then this behavior makes sense.
# 
# Note also that if we use "``=``" to assign another value to ``x``, this will not affect the value of ``y`` – assignment is simply a change of what object the variable points to:

# In[30]:


x = 'something else'
print(y)  # y is unchanged


# Again, this makes perfect sense if you think of x and y as pointers, and the "=" operator as an operation that changes what the name points to.
# 
# You might wonder whether this pointer idea makes arithmetic operations in Python difficult to track, but Python is set up so that this is not an issue. Numbers, strings, and other simple types are immutable: you can't change their value – you can only change what values the variables point to. So, for example, it's perfectly safe to do operations like the following:

# In[32]:


x = 10
y = x
x += 5  # add 5 to x's value, and assign it to x
print("x =", x)
print("y =", y)


# In[ ]:





# When we call ``x += 5``, we are not modifying the value of the ``10`` object pointed to by ``x``; we are rather changing the variable ``x`` so that it points to a new integer object with value ``15``.
# For this reason, the value of ``y`` is not affected by the operation.

# In[ ]:





# ## Arithmetic Operations
# Python implements seven basic binary arithmetic operators, two of which can double as unary operators.
# They are summarized in the following table:
# 
# | Operator     | Name           | Description                                            |
# |--------------|----------------|--------------------------------------------------------|
# | ``a + b``    | Addition       | Sum of ``a`` and ``b``                                 |
# | ``a - b``    | Subtraction    | Difference of ``a`` and ``b``                          |
# | ``a * b``    | Multiplication | Product of ``a`` and ``b``                             |
# | ``a / b``    | True division  | Quotient of ``a`` and ``b``                            |
# | ``a // b``   | Floor division | Quotient of ``a`` and ``b``, removing fractional parts |
# | ``a % b``    | Modulus        | Integer remainder after division of ``a`` by ``b``     |
# | ``a ** b``   | Exponentiation | ``a`` raised to the power of ``b``                     |
# | ``-a``       | Negation       | The negative of ``a``                                  |
# | ``+a``       | Unary plus     | ``a`` unchanged (rarely used)                          |
# 
# These operators can be used and combined in intuitive ways, using standard parentheses to group operations.
# For example:

# In[36]:


# addition, subtraction, multiplication
(4 + 8) * (6.5 - 3)


# In[ ]:





# In[ ]:





# ## Comparison Operations
# 
# Another type of operation which can be very useful is comparison of different values.
# For this, Python implements standard comparison operators, which return Boolean values ``True`` and ``False``.
# The comparison operations are listed in the following table:
# 
# | ``a == b``| ``a`` equal to ``b``      
# | ``a != b`` | ``a`` not equal to ``b``             
# | ``a < b``| ``a`` less than ``b``         
# | ``a > b``| ``a`` greater than ``b``             
# | ``a <= b``| ``a`` less than or equal to ``b``
# |``a >= b`` | ``a`` greater than or equal to ``b``
# 
# 
# 
# These comparison operators can be combined with the arithmetic and bitwise operators to express a virtually limitless range of tests for the numbers.
# For example, we can check if a number is odd by checking that the modulus with 2 returns 1:

# In[37]:


2 < 1


# In[38]:


# 25 is odd
25 % 2 == 1


# In[ ]:





# In[39]:


# check if a is between 15 and 30
a = 25
15 < a < 30


# In[ ]:





# ## Boolean Operations
# When working with Boolean values, Python provides operators to combine the values using the standard concepts of "and", "or", and "not".
# Predictably, these operators are expressed using the words ``and``, ``or``, and ``not``:

# In[15]:


x = 4
(x < 6) and (x > 2)


# In[16]:


(x > 10) or (x % 2 == 0)


# In[17]:


not (x < 6)


# Boolean algebra aficionados might notice that the XOR operator is not included; this can of course be constructed in several ways from a compound statement of the other operators.
# Otherwise, a clever trick you can use for XOR of Boolean values is the following:

# In[18]:


# (x > 1) xor (x < 10)
(x > 1) != (x < 10)


# These sorts of Boolean operations will become extremely useful when we begin discussing *control flow statements* such as conditionals and loops.
# 
# One sometimes confusing thing about the language is when to use Boolean operators (``and``, ``or``, ``not``), and when to use bitwise operations (``&``, ``|``, ``~``).
# The answer lies in their names: Boolean operators should be used when you want to compute *Boolean values (i.e., truth or falsehood) of entire statements*.
# Bitwise operations should be used when you want to *operate on individual bits or components of the objects in question*.

# In[ ]:




