<template>
    <div class="main-content">
        <div class="aoi breadcrumb">
            You are here: Home > Services > Space & Facilities
        </div>
        <div class="aoi key-aoi title">
            <h2>Seat Reservation</h2>
        </div>
        <div class="aoi key-aoi content">
            <p>Welcome to the seat reservation system. Please follow the steps below to complete a booking:</p>
            <p>
                1. This system is intended for enrolled students. Entry to the library requires campus card access or
                facial recognition.
            </p>
            <p>
                2. Click the “Start Reservation” button to enter the booking flow and choose the library zone you want.
            </p>
            <p>
                3. Choose your preferred date and time range. The system will then display available seats.
            </p>
            <p>
                4. Select a seat, review the reservation details, and click “Submit”.
            </p>
            <p>
                5. Your reservation details will be sent to your email inbox. Please check it promptly.
            </p>
            <p>
                <strong>Important Notes:</strong>
            </p>
            <p>
                - Please book in advance to avoid limited availability during peak hours.
            </p>
            <p>
                - If you need to cancel, please do so at least 24 hours before the reserved time.
            </p>
            <p>
                - The reservation system is available on weekdays from 7:00 to 22:00. Plan ahead for weekends and
                holidays.
            </p>
        </div>
        <div class="button-container">
            <button class="aoi key-aoi start-button" @click="navigateToReserve">Start Reservation</button>
        </div>
    </div>
</template>

<script setup>
import { useRouter } from 'vue-router';

const router = useRouter();

const navigateToReserve = () => {
    router.push('/reserve');
};
</script>

<style scoped>
.title {
    text-align: center;
}

.main-content {
    padding: 20px 5%;
}

.content {
    text-indent: 2em;
    font-size: 18px;
    line-height: 2;
}

.content p {
    margin: 0;
}

.button-container {
    text-align: center;
    margin-top: 30px;
}

.start-button {
    font-size: 24px;
    padding: 10px 20px;
    background-color: #3597ff;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s;
}

.start-button:hover {
    background-color: #66aaf2;
}
</style>