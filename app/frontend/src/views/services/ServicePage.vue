<template>
    <div class="main-box">
        <div class="sidebar">
            <div class="bar-title">
                <h2 class="aoi">Services</h2>
            </div>
            <el-menu :default-active="activeMenu" class="el-menu-vertical" @select="handleSelect"
                background-color="#f7f7f7" text-color="#333" active-text-color="#20a0ff" mode="vertical">
                <el-menu-item class="aoip" index="BookBorrow">Borrowing</el-menu-item>
                <el-menu-item class="aoip" index="CardProcess">Library Card Services</el-menu-item>
                <el-menu-item class="aoip" index="AncientRead">Rare Books Reading</el-menu-item>
                <el-menu-item class="aoip" index="DiscRequest">Companion Disc Access</el-menu-item>
                <el-menu-item class="aoip" index="DocumentTransfer">Document Delivery</el-menu-item>
                <el-menu-item class="aoip" index="TechSearch">Novelty Search</el-menu-item>
                <el-menu-item class="aoip" index="InfoTeaching">Information Literacy Instruction</el-menu-item>
                <el-menu-item class="aoip key-aoi" index="SeatReserve">Space & Facilities</el-menu-item>
                <el-menu-item class="aoip" index="VolunteerTeam">Volunteer Team</el-menu-item>
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

// initialize active menu from current route
const activeMenu = ref(route.name);

// handle menu selection
const handleSelect = (index) => {
    activeMenu.value = index;
    router.push({ name: index });
};

// keep active menu synced with route changes
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
    /* font size */
}
</style>