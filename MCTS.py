from anytree import Node , RenderTree
import numpy as np
from tqdm import tqdm
import copy

class MCTS_node:
    def __init__(self,capital,hold,ROI,LOB,OB,prior_prob):
        
        # Funds held by the investor
        self.capital = capital
        # stock hold
        self.hold = hold
        # num of selection 
        self.visit_count=0
        # The benefit that can be obtained by selecting the node.
        self.ROI=ROI
        # current Limit order book
        self.LOB = LOB
        # current known Order Book
        self.OB = OB
        # The probability of policy network
        self.prior_prob = prior_prob
        
class MCTS:

    def __init__(self,LOB,init_capital = 10000 ,init_hold = 0, UCB_constant = 5 ,num_simulation = 10):
        
        self.init_capital = init_capital
        self.UCB_constant = UCB_constant
        self.init_hold = init_hold
        self.var_state_prob = 0.2
        self.num_simulation = num_simulation
        self.tick = 0.5
        self.SetRoot(LOB)
        # one ticks:
        # probability of bidPri_up: 0.0069 , bidPri_down: 0.0026
        # probability of askPri_up: 0.0024 , askPri_down: 0.007
        # two ticks : 
        # probability of bidPri_up: 0.1029 , bidPri_down: 0.0026
        # probability of askPri_up: 0.0028 , askPri_down: 0.0946

        self.prob_list=[[[0.0069,0.0095],[0.0024,0.0031]],[[0.1029,0.1055],[0.0028,0.0974]]]
        
    def Run(self,epoches = 20):
        #
        for epoch in range(epoches):
            loop = tqdm(epoch)
            loop.set_description(f"[{epoch+1}/{epoches}]")
        
            leaf_node = self.Selection(self.root)
            chosen_node = self.Expansion(leaf_node= leaf_node)
            ROI = self.Simulation(leaf_node= chosen_node)
            self.BackPropagation(expand_node= chosen_node ,ROI = ROI)

    # create complete order book
    def CreateOB(self,LOB):
        
        OB = dict()
        for i in range(0,5):
            OB[LOB[i]]=LOB[i+5]
            OB[LOB[i+10]]=LOB[i+15]
        
        return OB
    
    def CreateLOB(self,LOB):
        
        new_LOB=list()
        LOB_bid = list()
        LOB_ask = list()
        
        for i in range(0,5):
            LOB_bid.extend([LOB[i],LOB[i+5]])
        new_LOB.append(LOB_bid)
        for i in range(10,15):
            LOB_ask.extend([LOB[i],LOB[i+5]])
        new_LOB.append(LOB_ask)
        
        return new_LOB
    
    # create root data
    def SetRoot(self,LOB):

        OB = self.CreateOB(LOB)
        LOB = self.CreateLOB(LOB)

        self.root = Node(MCTS_node(
            capital = self.init_capital,
            hold = self.init_hold,
            ROI=0,LOB=LOB,OB=OB,prior_prob=0
        ))

    # UCB formula 
    def UCB(self,node_info):
        
        return node_info.ROI + \
            self.UCB_constant*((node_info.prior_prob)/(1+node_info.visit_count))

    def Selection(self,node):
        
        cur_node = node
        # check whether it's a leaf node
        while not cur_node.is_leaf:
            UCB_list = list()
            # calculate UCB value
            for child_node in cur_node.children:
                UCB_list.append(self.UCB(child_node.name))
            # find the node with maximum UCB value
            maxUCB_child_node = cur_node.children[np.argmax(UCB_list)]
            # update the cur_node
            cur_node =maxUCB_child_node

        return cur_node
    
    def EvalPolicyNetwork(self,node):
        r =np.random.uniform(0,1)
        return [r,(1-r)]
    
    
    def UpdateLOB(self,cur_LOB,cur_OB,match_state):
        
    
        rand_prob = round(np.random.uniform(0,1),4)
        
        # check price spread ， larger than two ticks : 1 , otherwise : 0
        state = 0
        if (cur_LOB[0][0] - cur_LOB[1][0]) > self.tick :
            state = 1
        change =np.random.uniform(0.7,1.3)
        qty_change = np.random.uniform(1,1.3)
        
        # match_price : 0 ,match bid price 1 
        # match_price : 1 ,match ask price 1
    
        # price1 up 
        if self.prob_list[state][match_state][0] >= rand_prob :
            # remove previous AskBid5
            cur_LOB.pop()
            cur_LOB.pop()
            
            new_pri1 = cur_LOB[match_state][1]+self.tick
            # insert new AskBid1
            # if order book has value
            if cur_LOB.has_key(new_pri1) :
                cur_LOB.insert(0,cur_OB[new_pri1])
            else :
                qty = 1
                cur_LOB.insert(0,qty)
                cur_OB[new_pri1] = qty
            cur_LOB.insert(0,new_pri1)

        # price1 down 
        elif self.prob_list[state][match_state][1] >= rand_prob:
            # remove previous AskBid1
            cur_LOB.pop(0)
            cur_LOB.pop(0)

            new_pri5 = cur_LOB[match_state][-2]+self.tick
            cur_LOB.append(new_pri5)
            # insert new AskBid5
            # if order book has value
            if cur_LOB.has_key(new_pri5) :
                cur_LOB.append(cur_OB[new_pri5])
            else :
                qty =  1
                cur_LOB.append(qty)
                cur_OB[new_pri5] = qty
        else :

            cur_LOB[match_state][1] = int(cur_LOB[match_state][1] *change)
            cur_OB[cur_LOB[match_state][0]] = cur_LOB[match_state][1]

        cur_LOB[(match_state+1)%2][1] = int(cur_LOB[(match_state+1)%2][1]*qty_change)
        cur_OB[cur_LOB[(match_state+1)%2][0]] = cur_LOB[(match_state+1)%2][1]
        
        return cur_LOB , cur_OB

    def Expansion(self,leaf_node):

        # The probability of policy network output that input leaf node
        cur_prior_prob = self.EvalPolicyNetwork(leaf_node)
        # next match price , 0 : bidPri1 , 1 : askPri1
        match_state = np.random.randint(2)
        cur_LOB = leaf_node.name.LOB
        cur_OB = leaf_node.name.OB
        cur_capital = leaf_node.name.capital
        cur_hold = leaf_node.name.hold
        
        new_LOB , new_OB = self.UpdateLOB(cur_LOB=cur_LOB,cur_OB=cur_OB,match_state=match_state)
        # match at askPri1
        if match_state:
            # sell
            new_capital = cur_capital + cur_hold*cur_LOB[match_state][0]
            Node(MCTS_node(new_capital ,cur_hold, 0,new_LOB,new_OB, cur_prior_prob[1]),parent= leaf_node)
        # match at bidPri1
        else:
            # buy 
            new_hold = cur_hold + int(cur_capital/cur_LOB[match_state][0])
            Node(MCTS_node(cur_capital ,new_hold, 0,new_LOB,new_OB, cur_prior_prob[1]),parent= leaf_node)
            
        
        #capital,hold,ROI,LOB,OB,prior_prob
        Node(MCTS_node(cur_capital ,cur_hold, 0,new_LOB,new_OB, cur_prior_prob[0]),parent= leaf_node)
        return leaf_node.children[np.random.randint(2)]
    
    def Simulation(self,leaf_node):
        
        cur_root_node = copy.deepcopy(leaf_node)
        for _ in range(self.num_simulation):
            
            leaf_node = self.Selection(node = cur_root_node)
            chosen_node = self.Expansion(leaf_node)
            ROI = chosen_node.name.capital / self.init_capital
            self.BackPropagation(chosen_node,ROI)
        
        return  self.ChooseVisitNode(cur_root_node).ROI
    
    def BackPropagation(self,expand_node,ROI):
        
        cur_node = expand_node
        # check whether it's root node
        while not cur_node.is_root:
            # update node's ROI 
            cur_node.name.ROI += ROI
            # update node's visit count 
            cur_node.name.visit_count += 1
            # calculate average ROI
            cur_node.name.ROI /= cur_node.name.visit_count
            # move to father node
            cur_node = expand_node.ancestors[0]
    
    def ChooseVisitNode(self,node):
        
        visit_count_list = list()
        for i in node.children:
            #print(i.name.ROI)
            visit_count_list.append(i.name.visit_count)
        return node.children[np.argmax(visit_count_list)].name
