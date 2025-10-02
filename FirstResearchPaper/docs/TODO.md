# *You have started this project cause you have to justify your mother sacrifice*

1)~~ Build the model with multiple layer~~  
2) Reduce the weight expostion
    1) ~~Euclidean normalization --> euclidean is helps to keep the vector direction and reduce vector (pythogar theorom is key) [Y=u70ZpQH1muc]~~
    2) Implementation the euclidean weight to over code and test the result --> normalize process module ==> euclidean weight not reduce the votage and current
        1) ~~we figure only connected process will run throgh simualation. no paralle system~~
        2) Dummy port build in the weight normalize and connect with decorder to implement euclidean normalization
            0) ~~Started doing house keeping work(organize every file and folder)~~
            1) ~~we facing dead lock for while using dummy port --> its dead locking exactly at 60 time step --> weight normonizer not received the spike so we have setup the recv() method in the weight normolizer~~
    3) we have find way reduce votage and currect value
        1) ~~Reduce the value of learning rate and A_min and A_plus~~

3) Build the perfect visualization graph for result
    1) ~~Add plot_neuron_behaviour function on the utils module~~
    2) Visualizing the result and analyze the network behaviour
        ~~1) Weights taking larger time to visualize because of the neuron count~~
        
4) Still there no stable in the v and u value
    1) ~~Learn Visualization plot --> Heat map~~
    2) ~~Inducing the du and dv value makes network u and v stable~~ --> *THIS MAJOR MAIL STONE IN OUR RESEARCH*

5) Do research for changing parameter litle bit will not be affect the largely, Now du and dv change 0.1 its creating major impact in the u and v and neuron spike
    1) ~~Changing into fixed point model~~
6) Home static implementation
    1) Create competition by using lateral inhinition --> but not work --> increasing accumation

7) Adaptive threshold


8) Train the large image and test result