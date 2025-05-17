// models/message.js
const mongoose = require('mongoose');

const messageSchema = new mongoose.Schema({
    from: String,
    number: String, // <-- nuevo campo para guardar el número limpio
    body: String,
    timestamp: Date,
});

const Message = mongoose.model('Message', messageSchema);

module.exports = Message;
