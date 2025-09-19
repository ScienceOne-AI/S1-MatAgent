import numpy as np
import itertools
from toolkits.hea import evaluate_alloy_workflow
import logging
logger = logging.getLogger(__name__)

component_deltaH={}
component_gradients={}

def find_deltaH(component):
    global component_deltaH, component_gradients
    found=False
    deltaH1 = 99
    deltaH3 = 99
    for c in component_deltaH.keys():
        if set(c) == set(component):
            deltaH1 = component_deltaH[c][0]
            deltaH3 = component_deltaH[c][2]
            found=True
            break
    return found, deltaH1, deltaH3

def find_stored_info(component):
    global component_deltaH, component_gradients
    found=False
    activity=None
    gradients = None
    for c in component_deltaH.keys():
        if set(c) == set(component):
            activity = -component_deltaH[c][1]
            gradients = component_gradients[c]
            found=True
            break
    return found, activity, gradients

def get_activity_gradients(component, record=True):
    found, activity, gradients = find_stored_info(component)
    if not found or not record:
        d1,d2,d3 = evaluate_alloy_workflow(component)
        d1,d2,d3 = dict(d1), dict(d2), dict(d3)
        activity = -d2['delta_h_sum']
        component_deltaH[tuple(component)] = [d1['delta_h_sum'],d2['delta_h_sum'],d3['delta_h_sum']]
        gradients={}
        for e in d1['gradient_sum']:
            gradients[e] = -d2['gradient_sum'][e]
        if record:
            component_gradients[tuple(component)] = gradients
    return activity, gradients

def choose_element_by_gradient(component_elements, ori_gradients, can_be_Ni):
    g = 10000
    chosen_e = 0

    for i, e in enumerate(component_elements):
        if (can_be_Ni or e != 'Ni') and ori_gradients[e]<g:
            chosen_e = e
            g = ori_gradients[e]
    
    i = component_elements.index(chosen_e)

    return chosen_e, i

def order_by_periodic_table(component):
    element_order = {'Ni':0, 'Co':27, 'Ru':44, 'Pt':78, 'Fe':26, 'Mo':42, 'Cu':29, 'Pd':46, 'Ir':77, 'Rh':45}  # 'Ni':28
    component_sorted = sorted(component, key=lambda x: element_order.get(x[0], 255))
    return component_sorted

def change_step(component, max_ratio=5, f = None):
    component = order_by_periodic_table(component)
    
    ori_activity, ori_gradients = get_activity_gradients(component)
    if f is not None:
        print(f"current component: {component}, activity: {ori_activity}", file = f)
    logger.info(f"\ncurrent component: {component}, activity: {ori_activity}, activity gradient of each element:")
    logger.info(ori_gradients)
    candidate=['Ni', 'Co', 'Ru', 'Pt', 'Fe', 'Mo', 'Cu', 'Pd', 'Ir', 'Rh']  # 
    component_elements = [p[0] for p in component]
    ratio_Ni = 0
    for p in component:
        if p[0]=='Ni':
            ratio_Ni = p[1]
            break
    can_be_Ni = True
    if ratio_Ni==4:
        can_be_Ni = False
    else:
        for p in component:
            if p[0]!='Ni' and p[1]>=ratio_Ni:
                can_be_Ni = False
                break
    chosen_e, index = choose_element_by_gradient(component_elements, ori_gradients, can_be_Ni)
    logger.info(f'replace {chosen_e}')

    new_components = []
    
    if component[index][1]>=2:  # can reduce the ratio of the element
        for i,p in enumerate(component):
            new_component = component.copy()
            if i != index and p[1]+1<=max_ratio:  # the element to be increased is not the element to be decreased, and the increased element cannot exceed max_ratio
                if p[0]=='Ni' or ratio_Ni - ('Ni'==chosen_e) >= p[1] + 1:
                    new_component[i] = (p[0],p[1]+1)
                    new_component[index] = (chosen_e,component[index][1]-1)
                    new_component = order_by_periodic_table(new_component)
                    new_components.append(new_component)
                    
    if chosen_e != 'Ni':
        e_list = list(set(candidate)-set(component_elements))
        e_list.sort()
        for e in e_list:
            new_component = component.copy()
            new_component[index] = (e,component[index][1])
            new_component = order_by_periodic_table(new_component)
            new_components.append(new_component)

    #print(new_components)
    activitys = []

    for j, new_component in enumerate(new_components):
        activity, gradients = get_activity_gradients(new_component)
        if f is not None:
            print(f'候选组分：{new_component}，活性：{activity}', file=f)
        activitys.append(activity)
        logger.info(f"{j+1}. 候选组分：{new_component}, activity: {activity}, activity gradient of each element:")
        logger.info(gradients)
        logger.info('---------------------------\n')
    new_component = new_components[np.argmax(activitys)]
    logger.info(f'\nbest component in candidates: {new_component}, activity: {max(activitys)}')
    if max(activitys)<=ori_activity:
        logger.info('candidates are not better than original component, no change')
        return component, ori_activity, ori_activity
    logger.info('====================')
    return new_component, ori_activity, max(activitys)

def _component_to_name(component):
    name = ""
    for p in component:
        name += p[0]
        name += str(p[1]*4)
    return name


