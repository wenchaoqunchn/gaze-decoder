<template>
    <div class="main-box">
        <div class="step-box">
            <div class="step-container">
                <el-steps class="steps" :active="activeStep" finish-status="success" align-center>
                    <el-step class="aoi key-aoi" v-for="step in steps" :key="step.title" :title="step.title" />
                </el-steps>
            </div>
        </div>
        <div class="step-content">
            <router-view />
        </div>
        <div class="button-container">
            <button class="aoim" @click="prevStep">
                {{ activeStep === 0 ? 'Back to Main Site' : 'Previous' }}
            </button>
            <button class="aoim key-aoi" @click="nextStep">
                {{ activeStep === steps.length - 1 ? 'Submit' : 'Next' }}
            </button>
        </div>
    </div>
</template>

<script setup>
import { ref, watchEffect } from 'vue';
import { useRouter, useRoute } from 'vue-router';

const activeStep = ref(0);
const router = useRouter();
const route = useRoute();

const steps = [
    { title: 'Choose Floor' },
    { title: 'Choose Time' },
    { title: 'Choose Seat' },
    { title: 'Confirm Details' }
];

// update active step from route name
watchEffect(() => {
    switch (route.name) {
        case 'FloorSelect':
            activeStep.value = 0;
            break;
        case 'TimeSelect':
            activeStep.value = 1;
            break;
        case 'SeatSelect':
            activeStep.value = 2;
            break;
        case 'InfoConfirm':
            activeStep.value = 3;
            break;
        default:
            activeStep.value = 0;
            break;
    }
});

const nextStep = () => {
    if (activeStep.value < steps.length - 1) {
        activeStep.value++;
        navigateToStep(activeStep.value);
    } else if (activeStep.value === steps.length - 1) {
        router.push({ name: 'SessionDone' });
    }
};

const prevStep = () => {
    if (activeStep.value > 0) {
        activeStep.value--;
        navigateToStep(activeStep.value);
    } else {
        router.push({ name: 'BookBorrow' });
    }
};

const navigateToStep = (step) => {
    switch (step) {
        case 0:
            router.push({ name: 'FloorSelect' });
            break;
        case 1:
            router.push({ name: 'TimeSelect' });
            break;
        case 2:
            router.push({ name: 'SeatSelect' });
            break;
        case 3:
            router.push({ name: 'InfoConfirm' });
            break;
        default:
            break;
    }
};
</script>

<style scoped>
.step-content {
    flex: 5;
}

.step-container {
    width: 30vw;
}

.step-box {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
}

.main-box {
    background-color: rgba(255, 255, 255, 0.928);
    height: 96%;
    padding: 15px 200px;
    text-align: left;
    display: flex;
    flex-direction: column;
}

.steps {
    --color-process: #409EFF;
    --color-success: #409EFF;
    --color-title: #303133;
}

:deep(.steps .el-step__icon.is-text) {
    border-radius: 50%;
    border: 3px solid;
    width: 50px;
    height: 50px;
    border-color: inherit;
    font-size: 20px;
}

:deep(.steps .is-process .el-step__icon.is-text) {
    background-color: var(--color-process);
    color: white;
}

:deep(.steps .is-success .el-step__icon.is-text),
:deep(.steps .is-wait .el-step__icon.is-text) {
    border-width: 2px;
}

:deep(.steps .is-wait .el-step__icon-inner) {
    /* background-color: var(--color-process); */
    /* color: white; */
    font-weight: normal;
}

:deep(.steps .el-step__line) {
    background-color: rgb(181, 181, 181);
    margin-top: 11px;
    margin-left: 30px;
    margin-right: 30px;
}

:deep(.steps .is-success .el-step__line-inner) {
    border-width: 1px !important;
    width: 100% !important;
    transition-delay: 0ms !important;
}

:deep(.steps .el-step__head.is-process) {
    color: var(--color-process);
    border-color: var(--color-process);
}

:deep(.steps .el-step__title.is-process) {
    font-weight: bold;
    color: var(--color-process);
}

:deep(.steps .el-step__head.is-success) {
    color: var(--color-success);
    border-color: var(--color-success);
}

:deep(.steps .el-step__title.is-success) {
    color: var(--color-success);
}

.button-container {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}

button {
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
    border: none;
    border-radius: 5px;
    background-color: #409EFF;
    color: white;
    margin: 0 10px;
}

button:disabled {
    background-color: #ccc;
    cursor: default;
}

button:hover {
    background-color: rgb(121.3, 187.1, 255);
    /* 更深的蓝色 */
}
</style>