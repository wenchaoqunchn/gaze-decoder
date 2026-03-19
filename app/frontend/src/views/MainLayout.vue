<template>
    <el-container class="no-select" :class="['main-container', currentBackground]" >
        <el-header class="header">
            <MainHeader />
        </el-header>
        <el-main class="main-content">
            <router-view />
        </el-main>
    </el-container>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import MainHeader from '../components/MainHeader.vue';

// 定义响应式变量
const currentBackground = ref('');

// 设置背景
const setBackground = () => {
    const hour = new Date().getHours();
    currentBackground.value = hour >= 18 || hour < 6 ? 'bg-night' : 'bg-light';
};

// 在组件挂载时调用设置背景函数
onMounted(setBackground);
</script>

<style scoped>
.header {
    height: 150px;
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    padding: 10px 150px;
    background-color: #00000031;
    color: rgb(255, 255, 255);
}

.main-container {
    height: 100vh;
    width: 100vw;
    background-size: cover;
    background-position: center;
    transition: background-image 0.5s ease;
    /* 添加过渡效果 */
}

.bg-light {
    background-image: url("../assets/bg-night.png");
}

.bg-night {
    background-image: url("../assets/bg-night.png");
}

.main-content {
    padding: 0px 0;
    flex-grow: 1;
}

.no-select {
    user-select: none;
}

.no-select * {
    user-select: none;
}
</style>