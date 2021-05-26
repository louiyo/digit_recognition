from dlc_practical_prologue import *
from models import *
from time import time
import torch
from torch import optim, nn

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

#There are three different function that compute the test accuracy, as some of our networks do not output exactly the same way, therefore need
#different ways of computig it


#The networks using an auxilliary loss have two outputs, one being the actual numbers outputed via a 10 features channel that determines the number via a LogSoftMax function
#and another that gives out 1 feature, the probability of the output being one
def compute_err_auxlosses(model, test_input, test_classes, test_target, batches, mini_batch_size):

    correct_count_digit, all_count_digit = 0, 0
    correct_count_equal, all_count_equal = 0, 0
    
    if not batches:
    
        for img, label, target in zip(test_input, test_classes, test_target):   
            with torch.no_grad():
                log_probs_digits, probs_equality = model(img)


            probs = torch.exp(log_probs_digits)
            _, preds = torch.max(probs,dim=1)
            true_labels = label

            for predicted, groundtruth in zip(preds, true_labels):
                if(predicted == groundtruth):
                    correct_count_digit += 1
                all_count_digit += 1

            if((torch.sigmoid(probs_equality) > 0.5 and target == 1) or 
                       (torch.sigmoid(probs_equality) <= 0.5 and target == 0)):
                correct_count_equal += 1
            all_count_equal +=1
            
    else:

        for i in range(0, len(test_input), mini_batch_size):
        
            with torch.no_grad():
                log_probs_digits, probs_equality = model(test_input.narrow(0, i, mini_batch_size))

            probs = torch.exp(log_probs_digits)
            _, preds = torch.max(probs,dim=2)
            true_labels = test_classes.narrow(0, i, mini_batch_size)
            targets = test_target.narrow(0, i, mini_batch_size)
            
            
            
            for predicted, groundtruth in zip(preds, true_labels):
                if(predicted[0] == groundtruth[0]):
                    correct_count_digit += 1
                if(predicted[1] == groundtruth[1]):
                    correct_count_digit += 1
                all_count_digit += 2
               
            
            for prob_equality, target in zip(probs_equality.view(-1), targets):
                if((prob_equality >= 0.5 and target == 1) or 
                           (prob_equality < 0.5 and target == 0)):
                    correct_count_equal += 1
                all_count_equal +=1
    
    mod_accuracy_dig = correct_count_digit/all_count_digit
    mod_accuracy_cla = correct_count_equal/all_count_equal
            
    print("Number Of Images Tested =", all_count_digit)
    print("\nModel Accuracy =", mod_accuracy_dig, '\n\n')
    print("Number Of Inequalities tested =", all_count_equal)
    print("\nModel Accuracy =", mod_accuracy_cla)

    return mod_accuracy_dig, mod_accuracy_cla


#The Logic network has a comparison embedded, therefore has the same first output as the auxilliary losses networks, but the second one is directly 1 or 0,
#therefore does not need the final sigmoid to determine the probability
def compute_err_logic(model, test_input, test_classes, test_target, batches, mini_batch_size):


    correct_count_digit, all_count_digit = 0, 0
    correct_count_equal, all_count_equal = 0, 0
    
    if not batches:
    
    
        for img, label, target, i in zip(test_input, test_classes, 
                                                test_target, range(len(test_classes))):   
            with torch.no_grad():
                log_probs_digits, probs_equality = model(img)

            probs = torch.exp(log_probs_digits)
            _, preds = torch.max(probs,dim=1)
            true_labels = label

            for predicted, groundtruth in zip(preds, true_labels):
                if(predicted == groundtruth):
                    correct_count_digit += 1
                all_count_digit += 1


            if((torch.sigmoid(probs_equality) > 0.5 and target == 1) or 
                       (torch.sigmoid(probs_equality) <= 0.5 and target == 0)):
                correct_count_equal += 1
            all_count_equal +=1
            
            
    else:

        for i in range(0, len(test_input), mini_batch_size):
        
            with torch.no_grad():
                log_probs_digits, probs_equality = model(test_input.narrow(0, b, mini_batch_size))
            probs = torch.exp(log_probs_digits)
            _, preds = torch.max(probs,dim=2)
            true_labels = test_classes.narrow(0, b, mini_batch_size)
            targets = test_target.narrow(0, b, mini_batch_size)
            
            print(preds, true_labels)
            
            for predicted, groundtruth in zip(preds, true_labels):
                if(predicted[0] == groundtruth[0]):
                    correct_count_digit += 1
                if(predicted[1] == groundtruth[1]):
                    correct_count_digit += 1
                all_count_digit += 2
               
            for prob_equality, target in zip(probs_equality.view(-1), targets):
                float_target = target.to(torch.float32)
                float_prob = prob_equality.to(torch.float32)
                if(float_prob == float_target):
                    correct_count_equal += 1
                all_count_equal +=1
            
    mod_accuracy_dig = correct_count_digit/all_count_digit
    mod_accuracy_cla = correct_count_equal/all_count_equal    
        
    print("Number Of Images Tested =", all_count_digit)
    print("\nModel Accuracy =", mod_accuracy_dig, '\n\n')
    print("Number Of Inequalities tested =", all_count_equal)
    print("\nModel Accuracy =", mod_accuracy_cla)

    return all_count_digit, mod_accuracy_dig, all_count_equal, mod_accuracy_cla


#The classifier Network directly outputs the probablity of the class being one, therefore only needs this sigmoid and not the digits determination
def compute_err_classif(model, test_input, test_classes, test_target, batches, mini_batch_size):

    correct_count_equal, all_count_equal = 0, 0
    
    if not batches:
    
    
        for img, target, i in zip(test_input, test_target, range(len(test_classes))):   
            with torch.no_grad():
                probs_equality = model(img)

            if((probs_equality > 0.5 and target == 1) or 
                       (torch.sigmoid(probs_equality) <= 0.5 and target == 0)):
                correct_count_equal += 1
            all_count_equal +=1
            
            
    else:

        for i in range(0, len(test_input), mini_batch_size):
        
            with torch.no_grad():
                probs_equality = model(test_input.narrow(0, b, mini_batch_size))
            
            targets = test_target.narrow(0, b, mini_batch_size)
               
            for prob_equality, target in zip(probs_equality.view(-1), targets):
                if((torch.sigmoid(prob_equality) >= 0.5 and target == 1) or 
                           (torch.sigmoid(prob_equality) < 0.5 and target == 0)):
                    correct_count_equal += 1
                all_count_equal +=1

    mod_accuracy_cla = correct_count_equal/all_count_equal 

    print("Number Of Inequalities tested =", all_count_equal)
    print("\nModel Accuracy =", mod_accuracy_cla)

    return all_count_equal, mod_accuracy_cla


#With this function, we compute the performances of networks using an auxilliary loss over 10 trainings of the model
def compute_performances_auxilliary(model, criterion1, criterion2, train_input, train_classes, 
                                    train_target, test_input, test_classes, test_target, verbose = False,
                                    batches = False, mini_batch_size = 25, lr = 1e-4, mom = 0.95):
        
    dig_acc_sum = 0   
    cla_acc_sum = 0  
    
    print("Beginning evaluation of model...")
    
    for i in range(0, 10):

        model = model_train_auxlosses(model, criterion1, criterion2, train_input, 
                              train_classes, train_target, lr = lr, mom = mom, verbose = verbose)
        dig_temp, cla_temp = compute_err_auxlosses(model, test_input, test_classes, 
                              test_target, batches, mini_batch_size)
        print("Training ",i+1,"/10 complete")
        dig_acc_sum += dig_temp
        cla_acc_sum += cla_temp
    
    print("\nAverage Digit Recognition Test Accuracy: ", dig_acc_sum/10)
    print("Average Classification Test Accuracy: ", cla_acc_sum/10)


#This function has the same function as the auxilliary one, but only need one criterion
def compute_performances(model, criterion, train_input, train_classes, 
                        train_target, test_input, test_classes, test_target, verbose = False,
                        batches = False, mini_batch_size = 25, lr = 1e-4, mom = 0.95):


    dig_acc_sum = 0   
    cla_acc_sum = 0 

    print("Beginning evaluation of model...")
    
    for i in range(0, 10):
        model = model_train(model, criterion, train_input, 
                            train_target, lr = lr, mom = mom, verbose = verbose)
        dig_temp, cla_temp = compute_err_logic(model, test_input, test_classes,
                                               test_target, batches, mini_batch_size)
        dig_acc_sum += dig_temp
        cla_acc_sum += cla_temp
        print("Training ",i+1,"/10 complete")
        
    
    print("Average Digit Recognition Test Accuracy: ", dig_acc_sum/10)
    print("Average Classification Test Accuracy: ", cla_acc_sum/10)

    cla_acc_sum = 0 

    for i in range(0, 10):

        dig_temp, cla_temp = compute_err_classif(model, test_input, test_classes,
                                                 test_target, batches, mini_batch_size)
        cla_acc_sum += cla_temp
    
    print("Average Classification Test Accuracy: ", cla_acc_sum/10)

    return model


#This function trains a model with the given criterion
def model_train(model, criterion, train_input, train_target,
                batch_size = 25, lr=1e-4, mom = 0.95, verbose = False):

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)
    epochs = 25
    mini_batch_size = batch_size
    
    # ici on devrait avoir pour le convo logic un seul output a gérer par nllloss
    # cependant on a 2 outputs dans celui-ci
    
    for e in range(epochs):
        time0 = time()
        running_loss = 0
        for b in range(0, train_input.size(0), mini_batch_size):


            optimizer.zero_grad()
            output, _ = model(train_input.narrow(0, b, mini_batch_size))
            
            loss = criterion(output.view(-1, 1), train_target.narrow(0, b, mini_batch_size).to(torch.float32))
            
            loss.backward()
            optimizer.step()

            running_loss += loss
        if verbose:
            print("Epoch {} - Training loss: {}".format(e+1, running_loss/len(train_input)))
            print("\nTraining Time =", round(time()-time0, 2), "seconds")     
         
    return model


#This function trains a model that needs an auxilliary loss
def model_train_auxlosses(model, criterion1, criterion2, train_input, 
                          train_classes, train_target, batch_size = 25, lr=1e-4, mom = 0.95, verbose = False):

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=mom)
    epochs = 25

    mini_batch_size = batch_size

    for e in range(epochs):

        time0 = time()
        running_loss = 0
        
        for b in range(0, train_input.size(0), mini_batch_size):


            optimizer.zero_grad()
            output1, output2 = model(train_input.narrow(0, b, mini_batch_size))
            
            loss1 = criterion1(output1.view(-1, 10), train_classes.narrow(0, b, mini_batch_size).view(-1))
            loss2 = criterion2(output2.view(-1), train_target.narrow(0, b, mini_batch_size).to(torch.float32))

            w2 = 0.6

            loss = loss1 + loss2
            loss.backward()
            optimizer.step()



            running_loss += loss1.item() + loss2.item() * w2
        if verbose:
            print("Epoch {} - Training loss: {}".format(e+1, running_loss/len(train_input)))
            print("\nTraining Time =", round(time()-time0, 2), "seconds")
    return model