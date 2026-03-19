<template>
    <el-steps :active="activeStep" finish-status="success">
        <el-step v-for="step in steps" :key="step.title" :title="step.title" :description="step.description" />
    </el-steps>
</template>

<script setup>
import { ref } from 'vue';

const activeStep = ref(0);

const steps = [
    { title: '步骤 1', description: '描述 1' },
    { title: '步骤 2', description: '描述 2' },
    { title: '步骤 3', description: '描述 3' },
];

const nextStep = () => {
    if (activeStep.value < steps.length - 1) {
        activeStep.value++;
    }
};

const prevStep = () => {
    if (activeStep.value > 0) {
        activeStep.value--;
    }
};
</script>

<style scoped>
.step-content {
    margin: 20px 0;
}

.buttons {
    display: flex;
    justify-content: space-between;
}
</style>