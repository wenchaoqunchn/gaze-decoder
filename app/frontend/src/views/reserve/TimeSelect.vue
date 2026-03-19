<template>
    <div class="time-page">
        <div class="notice">
            <h3 class="aoi key-aoi">Booking Notes:</h3>
            <ul class="aoi key-aoi">
                <li><strong>Booking window:</strong> You may choose any date from today through the next seven days.
                    Please make sure your booking falls within library opening hours.</li>
                <li><strong>Opening hours:</strong> The library is open daily from <strong>7:30 AM</strong> to
                    <strong>10:30 PM</strong>. Reservations must stay within that range.
                </li>
                <li><strong>Continuous timeslots:</strong> Reservations must use one continuous block of time. Please
                    avoid gaps when choosing start and end times.</li>
            </ul>
        </div>
        <div class="datetime-picker">
            <div class="date-container">
                <div class="aoim key-aoi label">Reservation Date:</div>
                <div class="date-box">
                    <el-config-provider :locale="locale">
                        <el-date-picker class="aoim key-aoi dp" v-model="selectedDate" type="date"
                            placeholder="Select date" @change="handleDateChange" :default-value="new Date(1998, 12)"
                            size="large" :disabled-date="disabledDate" :editable="false" :popper-options="{
                                modifiers: [
                                    {
                                        name: 'flip',
                                        options: {
                                            fallbackPlacements: ['bottom'],
                                            allowedAutoPlacements: ['bottom'],
                                        }
                                    }
                                ]
                            }" />
                    </el-config-provider>
                </div>
            </div>
            <div class="time-selects">
                <div class="time-select">
                    <div class="aoim key-aoi label">Start Time:</div>
                    <el-time-select class="aoi key-aoi" v-model="startTime" placeholder="Select start time"
                        start="07:30" end="22:00" step="00:30" :min-time="minTime" @change="handleTimeChange" />
                </div>
                <div class="time-select">
                    <div class="aoim key-aoi label">End Time:</div>
                    <el-time-select class="aoi key-aoi" v-model="endTime" placeholder="Select end time" start="07:30"
                        end="22:30" step="00:01" :min-time="startTime" @change="handleTimeChange" />
                </div>
            </div>
        </div>
    </div>
</template>

<script setup>
import { ref, computed } from 'vue';
import { ElConfigProvider } from 'element-plus';
import en from 'element-plus/es/locale/lang/en';
import { useStore } from 'vuex';

const store = useStore();

const selectedDate = ref(store.state.selectedDate);
const startTime = ref(store.state.startTime);
const endTime = ref(store.state.endTime);
const locale = en;
const minTime = ref('07:29');

const disabledDate = (date) => {
    const today = new Date();
    const oneWeekLater = new Date();
    oneWeekLater.setDate(today.getDate() + 7);
    today.setHours(0, 0, 0, 0);

    return date > oneWeekLater || date < today;
};

const handleDateChange = (date) => {
    const today = new Date();
    const selected = new Date(date);
    store.commit('setSelectedDate', date);

    if (selected.setHours(0, 0, 0, 0) === today.setHours(0, 0, 0, 0)) {
        const now = new Date();
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        minTime.value = `${hours}:${minutes}`;
    } else {
        minTime.value = '07:29';
    }
};

const handleTimeChange = () => {
    store.commit('setStartTime', startTime.value);
    store.commit('setEndTime', endTime.value);
};
</script>

<style scoped>
.datetime-picker {
    display: flex;
    gap: 20px;
    padding: 20px;
}

.time-page {
    margin: 10px 150px;
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.072);
    padding: 20px 60px;
}

.date-container {
    flex: 1;
    display: flex;
    flex-direction: column;
}

.time-selects {
    flex: 1.5;
    display: flex;
    gap: 30px;
}

.label {
    margin: 10px 0;
    font-weight: bold;
    color: #333;
}

.time-select {
    flex: 1;
    display: flex;
    flex-direction: column;
}

.notice {
    padding: 5px 0;
    /* max-width: 600px; */
    margin: auto;
    border-bottom: 1px solid #33333338;
}

h3 {
    color: #333;
    margin: 5px 0;
}

ul {
    list-style-type: disc;
    margin-left: 20px;
    color: #555;
}

li {
    margin-bottom: 5px;
}

strong {
    color: #407fba;
}
</style>