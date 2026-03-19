<template>
    <div class="main-box">
        <div class="sidebar">
            <div class="bar-title">
                <h2 class="aoi">About the Library</h2>
            </div>
            <el-menu :default-active="activeMenu" class="el-menu-vertical" @select="handleSelect"
                background-color="#f7f7f7" text-color="#333" active-text-color="#20a0ff" mode="vertical">
                <el-menu-item class="aoip" index="LibIntro">Library Profile</el-menu-item>
                <el-menu-item class="aoip" index="LeaderSpeech">Director's Message</el-menu-item>
                <el-menu-item class="aoip" index="LibRule">Policies</el-menu-item>
                <el-menu-item class="aoip" index="ServiceTime">Opening Hours</el-menu-item>
                <el-menu-item class="aoip" index="ServiceOverview">Service Directory</el-menu-item>
                <el-menu-item class="aoip" index="LibLayout">Floor Layout</el-menu-item>
            </el-menu>
        </div>
        <div class="main-content">
            <router-view></router-view>
        </div>
    </div>
</template>

<script setup>
import { ref, watch } from 'vue';
import { useRouter, useRoute } from 'vue-router';

const router = useRouter();
const route = useRoute();

// initialize active menu from the current route name
const activeMenu = ref(route.name);
// handle menu selection
const handleSelect = (index) => {
    activeMenu.value = index;
    router.push({ name: index });
};

// keep active menu in sync with route changes
watch(() => route.name, (newName) => {
    activeMenu.value = newName;
});
</script>

<style scoped>
h2 {
    margin: 0;
}

.title {
    text-align: center;
}

.main-box {
    background-color: rgba(255, 255, 255, 0.928);
    height: 95%;
    padding: 20px 200px;
    text-align: left;
    display: flex;
}

.sidebar {
    flex: 1;
    border: 1px solid #ccc;
    height: fit-content;
}

.bar-title {
    padding: 20px 0;
    background-size: cover;
    background-position: center;
    background-image: url(../../assets/sidebar.png);
    text-align: center;
}

.main-content {
    flex: 3;
    padding: 0 5%;
}

.content {
    line-height: 1;
}

table {
    line-height: 0.5;
}

.el-menu-item {
    font-size: 18px;
    /* 设置字体大小 */
}
</style>