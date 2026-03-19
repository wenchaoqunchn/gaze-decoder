<template>
    <img src="../assets/logo.png" alt="Logo" class="aoi logo" />
    <el-tabs v-model="activeTab" :tab-bar-style="{ transition: 'none' }" class="aoi nav" @tab-click="handleTabClick">
        <el-tab-pane label="Home" name="HomePage"></el-tab-pane>
        <el-tab-pane label="About" name="Overview"></el-tab-pane>
        <el-tab-pane label="Resources" name="Resources"></el-tab-pane>
        <el-tab-pane label="Services" name="Services"></el-tab-pane>
        <el-tab-pane label="Contact Us" name="ContactUs"></el-tab-pane>
    </el-tabs>
    <div class="time aoi">
        <img src="../assets/clock.gif" alt="Clock" class="gif" />
        <div>Today's Opening Hours</div>
        <div>{{ openingHours }}</div>
    </div>
</template>

<script setup>
import { ref, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router';

const router = useRouter();
const route = useRoute();
const openingHours = '7:30 - 22:30';

// 根据当前路由初始化 activeTab
const activeTab = ref(getActiveTab(route.path));

// 获取活跃的选项卡名称
function getActiveTab(path) {
    // 定义每个主要路由的路径前缀
    const pathMap = {
        '/home': 'HomePage',
        '/overview': 'Overview',
        '/resources': 'Resources',
        '/services': 'Services',
        '/contact': 'ContactUs',
        '/reserve': 'Services'
    };

    // 检查路径是否在 pathMap 中
    for (const key in pathMap) {
        console.log("path:" + path)
        console.log("key:" + key)
        console.log(path.startsWith(key))
        if (path.startsWith(key)) {
            return pathMap[key];
        }
    }

    // 默认返回首页
    return 'HomePage';
}

// 点击选项卡的处理函数
const handleTabClick = (tab) => {
    const tabName = tab.props.name;
    console.log('Tab clicked:', tabName);
    router.push({ name: tabName });
};

// 监视路由变化，更新 activeTab
watch(() => route.path, (newPath) => {
    activeTab.value = getActiveTab(newPath);
});
</script>

<style scoped>
.gif {
    width: 30px;
    height: auto;
}

:deep(.el-tabs__item) {
    color: rgb(255, 255, 255);
    font-size: 22px;
    margin: 0 20px;
    /* font-weight: bold; */
    /* 添加过渡效果 */
}

/* 鼠标悬浮和选中状态下的样式 */
:deep(.el-tabs__item.is-active),
:deep(.el-tabs__item:hover) {
    color: rgb(255, 255, 255);
    /* 文字颜色变为白色 */

    /* 字体加粗 */
}

:deep(.el-tabs__item.is-active) {
    font-weight: bold;
}

:deep(.el-tabs__nav-wrap::after) {
    position: static !important;
}

/* 下划线颜色 */
:deep(.el-tabs__active-bar) {
    background-color: rgb(61, 187, 255);
}

:deep(.el-tabs__active-bar) {
    transition: none;
    /* 取消动画 */
}

.logo {
    margin-bottom: 40px;
}

.nav {
    margin-bottom: 23px;
}

.time {
    margin-bottom: 20px;
}
</style>