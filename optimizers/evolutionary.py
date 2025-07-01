# optimizers/evolutionary.py
import numpy as np

class EvolutionaryOptimizer:
    def __init__(self, action_dim, config, device):
        self.action_dim = action_dim
        self.population_size = config.EVO_POPULATION_SIZE
        self.mutation_rate = config.EVO_MUTATION_RATE
        self.crossover_rate = config.EVO_CROSSOVER_RATE
        self.elitism_count = config.EVO_ELITISM_COUNT

    def _create_individual(self):
        raw = np.random.rand(self.action_dim)
        return raw / np.sum(raw)

    def _crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.action_dim - 1)
            child1_raw = np.concatenate([parent1[:point], parent2[point:]])
            child2_raw = np.concatenate([parent2[:point], parent1[point:]])
            return child1_raw / np.sum(child1_raw), child2_raw / np.sum(child2_raw)
        return parent1, parent2

    def _mutate(self, individual):
        for i in range(self.action_dim):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.normal(0, 0.1)
        individual[individual < 0] = 0 # 确保权重非负
        return individual / np.sum(individual) # 重新归一化

    def run(self, evaluate_weights_fn, epochs, **kwargs):
        # 1. 初始化种群
        population = [self._create_individual() for _ in range(self.population_size)]
        
        best_overall_acc = 0.0
        best_overall_weights = None

        for generation in range(epochs):
            # 2. 评估适应度
            fitness = []
            for individual in population:
                acc, _ = evaluate_weights_fn(individual)
                fitness.append(acc)
            
            fitness = np.array(fitness)
            best_gen_idx = np.argmax(fitness)
            best_gen_acc = fitness[best_gen_idx]
            
            print(f"Generation {generation+1}/{epochs} [Evo] | Best Acc: {best_gen_acc:.2f}% | Avg Acc: {np.mean(fitness):.2f}%")

            if best_gen_acc > best_overall_acc:
                best_overall_acc = best_gen_acc
                best_overall_weights = population[best_gen_idx]

            # 3. 选择、交叉、变异，产生新一代
            new_population = []
            
            # 精英主义：直接保留最好的个体
            elite_indices = np.argsort(fitness)[-self.elitism_count:]
            for idx in elite_indices:
                new_population.append(population[idx])
            
            # 轮盘赌选择 + 交叉 + 变异
            fitness_sum = np.sum(fitness)
            selection_probs = fitness / fitness_sum if fitness_sum > 0 else None
            
            while len(new_population) < self.population_size:
                # 选择
                if selection_probs is not None:
                    parent1_idx, parent2_idx = np.random.choice(len(population), 2, p=selection_probs, replace=False)
                else: # 如果适应度全为0
                    parent1_idx, parent2_idx = np.random.choice(len(population), 2, replace=False)
                
                parent1 = population[parent1_idx]
                parent2 = population[parent2_idx]
                
                # 交叉
                child1, child2 = self._crossover(parent1, parent2)
                
                # 变异
                new_population.append(self._mutate(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self._mutate(child2))

            population = new_population

        return best_overall_acc, best_overall_weights