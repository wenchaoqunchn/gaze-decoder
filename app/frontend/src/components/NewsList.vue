<template>
    <div class="news-container">
        <h1 class="aoi header">Latest News</h1>
        <el-table class="aoi tab" :data="newsList">
            <el-table-column prop="date" label="Date" width="80">
                <template #default="{ row }">
                    <span class="date">{{ row.date }}</span>
                </template>
            </el-table-column>
            <el-table-column prop="type" label="Category" width="120">
                <template #default="{ row }">
                    <span :class="['type', getTypeClass(row.type)]">{{ row.type }}</span>
                </template>
            </el-table-column>
            <el-table-column prop="title" label="Title">
                <template #default="{ row }">
                    <span class="title">{{ row.title }}</span>
                </template>
            </el-table-column>
        </el-table>
    </div>
</template>

<script>
export default {
    name: 'NewsList',
    data() {
        return {
            newsList: [
                { date: '10-24', type: 'Lecture', title: 'Workshop Series: Smarter Reading Strategies for Graduate Research' },
                { date: '10-18', type: 'Announcement', title: 'Extended Evening Opening Hours Available During Midterm Review Week' },
                { date: '10-12', type: 'Announcement', title: 'Library Orientation Sessions Open for New Students and Visiting Scholars' },
                { date: '10-08', type: 'Resource', title: 'Updated Research Help Guide Released with New Search Tips and FAQs' },
                { date: '10-02', type: 'Trial', title: 'Temporary Access Now Available for an Interdisciplinary E-Journal Collection' },
                { date: '09-29', type: 'Announcement', title: 'Quiet Study Policy Reminder and Seat Booking Advice for Busy Hours' },
            ],
        };
    },
    methods: {
        getTypeClass(type) {
            switch (type) {
                case 'Lecture':
                    return 'lecture';
                case 'Announcement':
                    return 'announcement';
                case 'Resource':
                    return 'resource-recommendation';
                case 'Trial':
                    return 'resource-trial';
                default:
                    return '';
            }
        },
    },
};
</script>

<style scoped>
.news-container {
    padding: 20px;
}

.header {
    font-size: 36px;
    margin: 0;
}

.tab {
    background-color: white;
}

.date,
.title {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.type {
    display: inline-block;
    border-radius: 50px;
    padding: 0px 15px;
    color: white;
    font-size: 10px;
}

.lecture {
    background-color: rgb(30, 173, 198);
}

.announcement {
    background-color: rgb(241, 154, 5);
}

.resource-recommendation {
    background-color: rgb(88, 191, 19);
}

.resource-trial {
    background-color: rgb(85, 144, 233);
}
</style>