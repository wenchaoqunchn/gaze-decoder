<template>
    <div class="main-box">
        <div class="sidebar">
            <div class="bar-title">
                <h2 class="aoi">Contact Us</h2>
            </div>
        </div>
        <div class="main-content">
            <div class="title">
                <h1 class="aoi">Contact Us</h1>
            </div>
            <div class="aoi content">
                <p style="margin-top: 20px;"><strong>General Inquiries:</strong> +1-555-392-2185 ; +1-555-396-6505</p>
                <p style="margin-top: 5px;">
                    <strong>Academic Visits:</strong> +1-555-392-2063 (Please book at least 3 business days in advance
                    and send the visit request form to visit@library.edu)
                </p>
                <p style="margin-top: 5px;"><strong>Main Service Desk:</strong> +1-555-391-1819</p>
                <p style="margin-top: 5px;"><span style="color: red;"><strong>24-Hour Emergency Line<span
                                style="color: rgb(0, 0, 0);">:</span></strong></span> +1-555-396-6502</p>
                <p style="margin-top: 5px;"><strong>Volunteer Registration:</strong> +1-555-396-6516</p>

                <p style="text-align: center; margin-top: 20px; margin-bottom: 15px;"><strong>Department
                        Contacts</strong></p>
                <div align="center">
                    <table cellspacing="0" cellpadding="0" border="1">
                        <tbody>
                            <tr class="firstRow">
                                <td width="184" align="center" valign="middle">
                                    <p><strong>Department</strong></p>
                                </td>
                                <td width="220" align="center" valign="middle">
                                    <p><strong>Phone</strong></p>
                                </td>
                                <td width="206.66666666666666" align="center" valign="middle">
                                    <p><strong>Email</strong></p>
                                </td>
                            </tr>
                            <tr>
                                <td width="184" align="center" valign="middle">
                                    <p>General Office</p>
                                </td>
                                <td width="220.66666666666666" align="center" valign="middle">
                                    <p>+1-555-392-2063</p>
                                    <p>+1-555-392-2188</p>
                                </td>
                                <td width="207.66666666666666" align="center" valign="middle">
                                    <p>office@library.edu</p>
                                </td>
                            </tr>
                            <tr>
                                <td width="184" align="center" valign="middle">
                                    <p>Reference Services</p>
                                </td>
                                <td width="220" align="center" valign="middle">
                                    <p>+1-555-392-2185</p>
                                    <p>+1-555-396-6687</p>
                                </td>
                                <td width="208.66666666666666" align="center" valign="middle">
                                    <p>reference@library.edu</p>
                                </td>
                            </tr>
                            <tr>
                                <td width="184" align="center" valign="middle">
                                    <p>Acquisitions</p>
                                </td>
                                <td width="220" align="center" valign="middle">
                                    <p>+1-555-391-1820</p>
                                    <p>+1-555-396-6235</p>
                                </td>
                                <td width="209.66666666666666" align="center" valign="middle">
                                    <p>acquisitions@library.edu</p>
                                </td>
                            </tr>
                            <tr>
                                <td width="184" align="center" valign="middle">
                                    <p>Reader Services</p>
                                </td>
                                <td width="220" align="center" valign="middle">
                                    <p>+1-555-391-1822</p>
                                    <p>+1-555-396-6518</p>
                                </td>
                                <td width="211.66666666666666" align="center" valign="middle">
                                    <p>readerservices@library.edu</p>
                                </td>
                            </tr>
                            <tr>
                                <td width="184" align="center" valign="middle">
                                    <p>Engagement & Outreach</p>
                                </td>
                                <td width="220" align="center" valign="middle">
                                    <p>+1-555-396-6516</p>
                                </td>
                                <td width="211.66666666666666" align="center" valign="middle">
                                    <p>engagement@library.edu</p>
                                </td>
                            </tr>
                            <tr>
                                <td width="184" align="center" valign="middle">
                                    <p>Technology Services</p>
                                </td>
                                <td width="220" align="center" valign="middle">
                                    <p>+1-555-396-6501</p>
                                </td>
                                <td width="212.66666666666666" align="center" valign="middle">
                                    <p class="vsbcontent_end">tech@library.edu</p>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

</template>

<script>
export default {
    name: 'ContactUs',
};
</script>

<style scoped>
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
}

.bar-title {
    background-size: cover;
    background-position: center;
    background-image: url(../assets/sidebar.png);
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
</style>