<template>
    <div class="main-content">
        <div class="aoi breadcrumb">
            You are here: Home > About the Library > Opening Hours
        </div>
        <div class="aoi title">
            <h2>Opening Hours</h2>
        </div>
        <div class="aoi content">
            <table class="service-table">
                <thead>
                    <tr>
                        <th>Location</th>
                        <th>Service</th>
                        <th>Hours</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td rowspan="6">Main Campus</td>
                        <td>Reading Areas</td>
                        <td>Monday to Sunday 7:30–22:30</td>
                    </tr>
                    <tr>
                        <td>Staffed Circulation Desk</td>
                        <td>Monday to Friday 8:00–21:00<br>Saturday and Sunday 8:30–11:30; 13:30–16:30</td>
                    </tr>
                    <tr>
                        <td>Compact Stacks</td>
                        <td>Monday to Sunday 8:00–17:00</td>
                    </tr>
                    <tr>
                        <td>Study Hall</td>
                        <td>Monday to Sunday 7:00–23:00 (temporary changes may apply during exam periods)</td>
                    </tr>
                    <tr>
                        <td>Discussion Rooms</td>
                        <td>Monday to Sunday 8:00–21:00</td>
                    </tr>
                    <tr>
                        <td>Media Room</td>
                        <td>Monday to Sunday 8:00–22:00</td>
                    </tr>
                    <tr>
                        <td>Zhonglan Residence Branch</td>
                        <td colspan="2">8:00–23:00</td>
                    </tr>
                </tbody>
            </table>

            <p>Notes:</p>
            <ol>
                <li>Self-service borrowing and returns are available at the main campus during open hours.</li>
                <li>The library is closed on national public holidays.</li>
            </ol>
        </div>
    </div>
</template>

<script setup>
</script>

<style scoped>
.title {
    text-align: center;
}

.main-content {
    padding: 20px 5%;
}

.content {
    font-size: 18px;
    line-height: 1.6;
}

.service-table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}

.service-table th,
.service-table td {
    border: 1px solid #ccc;
    padding: 10px;
    text-align: left;
}

.service-table th {
    background-color: #f2f2f2;
}
</style>