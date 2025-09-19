import random
import math
from toolkits.hea import evaluate_alloy_workflow
random.seed(42)

metal_elements = {'Co', 'Ru', 'Pt', 'Fe', 'Mo', 'Cu', 'Pd', 'Ir', 'Rh'}


def find_combinations(n):
    s = n - 5
    if s < 0:
        return []

    combinations = []
    min_atoms = max(1, math.ceil(0.05 * n))  # 每个元素最少原子数（至少5%，但不少于1个原子）
    max_atoms = int(0.35 * n)  # 每个元素最多原子数（不超过35%）

    def dfs(remaining, path):
        if len(path) == 4:
            # 检查最后一个元素是否在范围内
            last_element = remaining + 1  # +1是因为后面会对所有元素+1
            if min_atoms <= last_element <= max_atoms:
                combo = path + (remaining,)
                t = tuple(x + 1 for x in combo)
                if t[0] >= t[1] and t[0] >= t[2] and t[0] >= t[3] and t[0] >= t[4]:
                    combinations.append(t)
            return

        # 遍历可能的原子数
        for i in range(max(0, min_atoms - 1), min(remaining + 1, max_atoms)):
            # i+1将是实际原子数（因为后面会对所有元素+1）
            if i + 1 >= min_atoms and i + 1 <= max_atoms:
                dfs(remaining - i, path + (i,))

    dfs(s, ())
    return combinations


def reorder_lists_by_order(list1, list2, order):
    # 提取元素符号（元组的第一个元素）
    elements1 = [x[0] for x in list1]
    elements2 = [x[0] for x in list2]

    # 找出匹配的元素（在两个列表中都存在的元素）
    common_elements = set(elements1) & set(elements2)

    # 找出仅出现在 list1 或 list2 的元素
    only_in_list1 = [x for x in elements1 if x not in common_elements]
    only_in_list2 = [x for x in elements2 if x not in common_elements]

    # 按给定的 order 排序匹配项
    common_sorted = sorted(common_elements, key=lambda x: order.index(x) if x in order else float('inf'))

    # 按给定的 order 排序仅出现在 list1 或 list2 的元素
    only_in_list1_sorted = sorted(only_in_list1, key=lambda x: order.index(x) if x in order else float('inf'))
    only_in_list2_sorted = sorted(only_in_list2, key=lambda x: order.index(x) if x in order else float('inf'))

    # 构建新的顺序：匹配项 + 仅 list1 项 + 仅 list2 项（但这里需要分别处理 list1 和 list2）
    # 注意：这里我们分别处理 list1 和 list2，因为它们的独有项可能不同
    new_order_list1 = common_sorted + only_in_list1_sorted
    new_order_list2 = common_sorted + only_in_list2_sorted

    # 重新构建 list1 和 list2
    # 使用字典提高查找效率
    dict1 = {x[0]: x for x in list1}
    dict2 = {x[0]: x for x in list2}

    # 按 new_order_list1 和 new_order_list2 重新排序
    reordered_list1 = [dict1[elem] for elem in new_order_list1]
    reordered_list2 = [dict2[elem] for elem in new_order_list2]

    return reordered_list1, reordered_list2


def check_component(component):
    elements = [p[0] for p in component]
    ratios = [p[1] for p in component]
    if 'Ni' not in elements:
        return False
    for e in elements:
        if e not in metal_elements | {'Ni'}:
            return False
    min_val = max(1, math.ceil(0.05 * sum(ratios)))
    max_val = math.floor(0.35 * sum(ratios))
    ni_index = elements.index('Ni')
    for r in ratios:
        if type(r) is not int or r<min_val or r>max_val or r>ratios[ni_index]:
            return False
    return True


def order_by_periodic_table(component):
    element_order = {'Ni':0, 'Co':27, 'Ru':44, 'Pt':78, 'Fe':26, 'Mo':42, 'Cu':29, 'Pd':46, 'Ir':77, 'Rh':45}  # 'Ni':28
    component_sorted = sorted(component, key=lambda x: element_order.get(x[0], 255))
    return component_sorted


class GeneticAlgorithm:
    def __init__(self, population_size=20, mutation_rate=0.2, crossover_rate=0.8, generations=50):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.valid_ratios = find_combinations(16)
        self.component_fitness = {}

    def find_min_manhattan(self, ratios):
        """
        在valid列表中找到与五元组a曼哈顿距离最小的五元组

        参数:
        valid -- 合法五元组的列表
        a -- 待比较的五元组

        返回:
        曼哈顿距离最小的五元组
        """
        # 计算缩放因子
        total_ratio = sum(ratios)
        scale = 16 / total_ratio
        scaled_ratios = [r * scale for r in ratios]

        min_distance = float('inf')
        result = None

        for candidate in self.valid_ratios:
            # 计算曼哈顿距离
            distance = sum(abs(scaled_ratios[i] - candidate[i]) for i in range(5))

            # 更新最小距离和结果
            if distance < min_distance:
                min_distance = distance
                result = candidate

        return result

    def initialize_population(self):
        '''初始化种群'''
        population = []
        for _ in range(self.population_size):
            # 随机选择5种金属元素
            elements = ['Ni']+random.sample(list(metal_elements), 4)
            # 随机生成合法的元素比例
            legal_combinations = find_combinations(16)
            position = random.choice(legal_combinations)
            component = [(e, r) for e, r in zip(elements, position)]
            component = order_by_periodic_table(component)
            population.append(component)
        return population

    def evaluate(self, component):
        '''评估函数'''
        if tuple(component) in self.component_fitness.keys():
            return self.component_fitness[tuple(component)]
        
        try:
            d1,d2,d3 = evaluate_alloy_workflow(component)
            self.component_fitness[tuple(component)] = -d2['delta_h_sum']
            return -d2['delta_h_sum']
        except Exception as e:
            print(f"评估错误: {e}")
            return 0




    def select_parents(self, population, fitness_values):
        '''使用轮盘赌选择父代'''
        total_fitness = sum(max(0.01, f) for f in fitness_values)  # 避免所有适应度都为0
        selection_probs = [max(0.01, f) / total_fitness for f in fitness_values]

        parents = []
        for _ in range(2):
            r = random.random()
            cumulative_prob = 0
            for i, prob in enumerate(selection_probs):  # 例如将0-1划分为0-0.2/0.2-0.5/0.5-1，看rand落在何区间
                cumulative_prob += prob
                if r <= cumulative_prob:
                    parents.append(population[i])
                    break
        return parents

    def crossover(self, parent1, parent2, f):
        '''交叉操作'''
        if random.random() > self.crossover_rate and parent1 != parent2:
            return parent1, parent2
        # 匹配
        elements = ['Co', 'Ru', 'Pt', 'Fe', 'Mo', 'Cu', 'Pd', 'Ir', 'Rh']
        random.shuffle(elements)
        order = ['Ni'] + elements
        parent1, parent2 = reorder_lists_by_order(parent1, parent2, order)
        print(f'匹配后：{parent1}、{parent2}', file=f)
        # 随机选择交叉点
        crossover_point = random.randint(2, 4)

        # 交换交叉点后的元素
        part1, part2 = parent1[crossover_point:], parent2[crossover_point:]
        new_1e = [p[0] for p in parent1[:crossover_point]] + [p[0] for p in part2]
        new_2e = [p[0] for p in parent2[:crossover_point]] + [p[0] for p in part1]
        sum_part1r = sum([p[1] for p in part1])
        sum_part2r = sum([p[1] for p in part2])
        _new_1r = [p[1] for p in parent1[:crossover_point]] + [p[1]*sum_part1r/sum_part2r for p in part2]
        _new_2r = [p[1] for p in parent2[:crossover_point]] + [p[1]*sum_part2r/sum_part1r for p in part1]

        new_1r = self.find_min_manhattan(_new_1r)
        new_2r = self.find_min_manhattan(_new_2r)
        new1 = [(e, r) for e, r in zip(new_1e, new_1r)]
        new2 = [(e, r) for e, r in zip(new_2e, new_2r)]

        return new1, new2

    def mutate(self, component):
        '''变异操作'''
        if round(random.random(),3) > self.mutation_rate:  # 防止浮点数误差影响复现
            return component

        # 随机选择变异类型
        mutation_type = random.choice(['element', 'ratio'])
        mutated_component = component.copy()

        if mutation_type == 'element':
            # 随机替换一个元素
            index = random.randint(1, 4)
            new_element = random.choice(list(metal_elements-{p[0] for p in mutated_component}))
            mutated_component[index] = (new_element, mutated_component[index][1])
        else:
            # 随机调整元素比例
            neighbours = []
            for i in range(1, 5):
                for j in range(1, 5):
                    if i != j:
                        mutated_component = component.copy()
                        mutated_component[i] = (mutated_component[i][0], mutated_component[i][1]-1)
                        mutated_component[j] = (mutated_component[j][0], mutated_component[j][1]+1)
                        if check_component(mutated_component):
                            neighbours.append(mutated_component)
            if neighbours:
                mutated_component = random.choice(neighbours)

        return mutated_component

    def run(self, file_path, population=None):
        '''运行遗传算法'''
        f = open(file_path, mode='a', encoding='utf-8')
        max_fitness_idx = 0
        # 初始化种群
        if population is None:
            population = self.initialize_population()
            init_population = tuple(population)

        for generation in range(self.generations):
            print(f"第 {generation + 1} 代", file=f)

            # 评估种群
            fitness_values = [self.evaluate(component) for component in population]
            for i,p in enumerate(population):
                print(f'{p} 适应度：{fitness_values[i]}', file=f)
            # 找到最佳解
            max_fitness_idx = fitness_values.index(max(fitness_values))
            if fitness_values[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitness_values[max_fitness_idx]
                self.best_solution = population[max_fitness_idx]
                print(
                    f"新的最佳解：{self.best_solution} 适应度：{self.best_fitness}", file=f)
            else:
                print(f'本轮最佳适应度：{max(fitness_values)} 比不上当前最佳适应度：{self.best_fitness}', file=f)
                break

            # 创建新一代
            new_population = []

            # 精英保留策略 - 保留最佳个体
            new_population.append(population[max_fitness_idx])
            print(f'保留{population[max_fitness_idx]}', file=f)
            # 生成新个体
            while len(new_population) < self.population_size:
                # 选择父代
                min_fitness = min(fitness_values)
                parent1, parent2 = self.select_parents(population, [f-min_fitness for f in fitness_values])
                while parent1==parent2:
                    parent1, parent2 = self.select_parents(population, [f-min_fitness for f in fitness_values])
                print(f'选择{parent1}+{parent2}', file=f)
                # 交叉
                child1, child2 = self.crossover(parent1, parent2, f)
                print(f'交叉后：{child1}、{child2}', file=f)

                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                print(f'变异后：{child1}、{child2}', file=f)
                child1 = order_by_periodic_table(child1)
                child2 = order_by_periodic_table(child2)
                # 添加到新种群
                new_population.append(child1)
                
                if len(new_population) < self.population_size:
                    new_population.append(child2)
                    print(f'添加{child1}、{child2}', file=f)
                else:
                    print(f'添加{child1}', file=f)

            # 更新种群
            population = new_population

        print("\n遗传算法搜索完成", file=f)
        print(f"最佳解: {self.best_solution}, 适应度：{self.best_fitness}", file=f)
        return self.best_solution, init_population


if __name__ == '__main__':
    population = [[('Ni', 5), ('Rh', 4), ('Pd', 5), ('Ir', 1), ('Pt', 1)],
[('Ni', 4), ('Cu', 4), ('Mo', 1), ('Ir', 4), ('Pt', 3)],
[('Ni', 4), ('Mo', 3), ('Rh', 4), ('Pd', 2), ('Pt', 3)],
[('Ni', 4), ('Fe', 2), ('Mo', 3), ('Ru', 3), ('Pd', 4)],
[('Ni', 5), ('Co', 4), ('Cu', 1), ('Mo', 4), ('Pd', 2)],
[('Ni', 5), ('Cu', 2), ('Rh', 1), ('Pd', 4), ('Pt', 4)],
[('Ni', 5), ('Fe', 3), ('Cu', 2), ('Mo', 4), ('Rh', 2)],
[('Ni', 5), ('Cu', 4), ('Mo', 2), ('Ru', 4), ('Pd', 1)],
[('Ni', 5), ('Fe', 1), ('Ru', 3), ('Ir', 2), ('Pt', 5)],
[('Ni', 5), ('Mo', 1), ('Pd', 1), ('Ir', 5), ('Pt', 4)],]
    ga = GeneticAlgorithm(population_size=10)
    f=open('/home/ubuntu/Desktop/projects/owl/tmp/genetic_seed42.txt','w')
    f.close()
    best_solution,_ = ga.run(file_path = '/home/ubuntu/Desktop/projects/owl/tmp/genetic_seed42.txt', population=population)#

